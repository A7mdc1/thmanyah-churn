"""
Model training and evaluation for customer churn prediction.

Implements proper time-based splits, class imbalance handling, and
business-relevant metrics.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MLFLOW_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """Trains and evaluates churn prediction models with proper validation."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)

        self.models = {
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "random_forest": RandomForestClassifier(
                random_state=42, class_weight="balanced"
            ),
            "logistic_regression": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
        }

        self.best_model = None
        self.best_score = 0
        self.results = {}

    def create_time_based_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple:
        """Create time-based train/test split to prevent data leakage."""
        # Sort by user and timestamp
        df_sorted = df.sort_values(["userId", "days_since_registration"])

        # Use temporal split based on days since registration
        cutoff = df_sorted["days_since_registration"].quantile(1 - test_size)

        train_mask = df_sorted["days_since_registration"] <= cutoff
        test_mask = df_sorted["days_since_registration"] > cutoff

        train_df = df_sorted[train_mask]
        test_df = df_sorted[test_mask]

        logger.info(f"Train set: {len(train_df)} users, Test set: {len(test_df)} users")
        logger.info(f"Train churn rate: {train_df['is_churned'].mean():.3f}")
        logger.info(f"Test churn rate: {test_df['is_churned'].mean():.3f}")

        return train_df, test_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple:
        """Prepare features and target for modeling."""
        # Select feature columns (exclude identifiers and targets)
        exclude_cols = [
            "userId",
            "is_churned",
            "churn_type",
            "churn_timestamp",
            "location_state",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle categorical location_state if present
        if "location_state" in df.columns:
            # One-hot encode top states, others as 'Other'
            top_states = df["location_state"].value_counts().head(10).index
            for state in top_states:
                df[f"state_{state}"] = (df["location_state"] == state).astype(int)
                feature_cols.append(f"state_{state}")

        X = df[feature_cols].fillna(0)
        y = df["is_churned"].astype(int)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y, feature_cols

    def evaluate_model(self, model, X_test, y_test) -> Dict:
        """Evaluate model with business-relevant metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        f1 = f1_score(y_test, y_pred)
        precision_score_val = precision_score(y_test, y_pred)
        recall_score_val = recall_score(y_test, y_pred)

        # Recall at different thresholds (business-relevant)
        recall_at_90_precision = 0
        recall_at_80_precision = 0

        for i, p in enumerate(precision):
            if p >= 0.9 and recall_at_90_precision == 0:
                recall_at_90_precision = recall[i]
            if p >= 0.8 and recall_at_80_precision == 0:
                recall_at_80_precision = recall[i]

        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "precision": precision_score_val,
            "recall": recall_score_val,
            "recall_at_90_precision": recall_at_90_precision,
            "recall_at_80_precision": recall_at_80_precision,
        }

        return metrics

    def train_model(self, model_name: str, X_train, y_train, X_test, y_test) -> Dict:
        """Train a single model with class imbalance handling."""
        logger.info(f"Training {model_name}")

        # Create pipeline with scaling and model (using class_weight for imbalance)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("classifier", self.models[model_name])]
        )

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics = self.evaluate_model(pipeline, X_test, y_test)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(pipeline.named_steps["classifier"], "feature_importances_"):
            feature_importance = pipeline.named_steps["classifier"].feature_importances_
        elif hasattr(pipeline.named_steps["classifier"], "coef_"):
            feature_importance = np.abs(pipeline.named_steps["classifier"].coef_[0])

        result = {
            "model": pipeline,
            "metrics": metrics,
            "feature_importance": feature_importance,
        }

        pr_auc = metrics["pr_auc"]
        f1_score = metrics["f1_score"]
        logger.info(f"{model_name} - PR-AUC: {pr_auc:.3f}, F1: {f1_score:.3f}")

        return result

    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train all models and return results."""
        # Set up MLflow tracking
        mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
        mlflow.set_experiment("thmanyah-churn")

        # Create time-based split
        train_df, test_df = self.create_time_based_split(df)

        # Prepare features
        X_train, y_train, feature_cols = self.prepare_features(train_df)
        X_test, y_test, _ = self.prepare_features(test_df)

        # Ensure same features in train and test
        common_features = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_features]
        X_test = X_test[common_features]

        logger.info(f"Using {len(common_features)} features for modeling")

        # Train all models with MLflow tracking
        results = {}
        for model_name in self.models.keys():
            with mlflow.start_run(run_name=f"churn_{model_name}"):
                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("split_strategy", "time_based")
                mlflow.log_param("train_size", len(train_df))
                mlflow.log_param("test_size", len(test_df))
                mlflow.log_param("n_features", len(common_features))
                mlflow.log_param("class_balancing", "balanced_weights")

                results[model_name] = self.train_model(
                    model_name, X_train, y_train, X_test, y_test
                )

                # Log metrics
                metrics = results[model_name]["metrics"]
                mlflow.log_metrics(metrics)

                # Log model
                model_pipeline = results[model_name]["model"]
                mlflow.sklearn.log_model(model_pipeline, "model")

        # Select best model based on PR-AUC
        best_model_name = max(
            results.keys(), key=lambda k: results[k]["metrics"]["pr_auc"]
        )
        self.best_model = results[best_model_name]["model"]
        self.best_score = results[best_model_name]["metrics"]["pr_auc"]

        logger.info(f"Best model: {best_model_name} with PR-AUC: {self.best_score:.3f}")

        # Save best model
        model_path = self.models_dir / f"best_churn_model_{best_model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Saved best model to {model_path}")

        self.results = results
        return results

    def error_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform error analysis on model predictions."""
        if not self.best_model:
            raise ValueError("No trained model found. Call train_all_models first.")

        # Split data again for analysis
        train_df, test_df = self.create_time_based_split(df)
        X_test, y_test, _ = self.prepare_features(test_df)

        # Ensure same features as training
        if hasattr(self, "results"):
            common_features = list(
                self.best_model.named_steps["scaler"].feature_names_in_
            )
            X_test = X_test[common_features]

        # Get predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # Create analysis dataframe
        analysis_df = test_df.copy()
        analysis_df["predicted_churn"] = y_pred
        analysis_df["churn_probability"] = y_pred_proba
        analysis_df["prediction_error"] = (y_pred != y_test).astype(int)

        # False positives and false negatives
        false_positives = analysis_df[
            (analysis_df["is_churned"] == 0) & (analysis_df["predicted_churn"] == 1)
        ]
        false_negatives = analysis_df[
            (analysis_df["is_churned"] == 1) & (analysis_df["predicted_churn"] == 0)
        ]

        error_analysis = {
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
            "false_positive_characteristics": {
                "avg_total_events": (
                    false_positives["total_events"].mean()
                    if len(false_positives) > 0
                    else 0
                ),
                "avg_days_since_registration": (
                    false_positives["days_since_registration"].mean()
                    if len(false_positives) > 0
                    else 0
                ),
                "paid_users_pct": (
                    (false_positives["current_level_paid"].mean() * 100)
                    if len(false_positives) > 0
                    else 0
                ),
            },
            "false_negative_characteristics": {
                "avg_total_events": (
                    false_negatives["total_events"].mean()
                    if len(false_negatives) > 0
                    else 0
                ),
                "avg_days_since_registration": (
                    false_negatives["days_since_registration"].mean()
                    if len(false_negatives) > 0
                    else 0
                ),
                "paid_users_pct": (
                    (false_negatives["current_level_paid"].mean() * 100)
                    if len(false_negatives) > 0
                    else 0
                ),
            },
        }

        return error_analysis


def main():
    """Main training pipeline."""
    from .churn_definition import ChurnDefinition
    from .data import ChurnDataLoader
    from .feature_engineering import ChurnFeatureEngineer

    # Load and prepare data
    loader = ChurnDataLoader()
    df = loader.load_event_logs()
    df = loader.basic_preprocessing(df)

    # Create features (using cutoff to simulate realistic scenario)
    feature_engineer = ChurnFeatureEngineer(prediction_cutoff="2018-11-15")
    features_df = feature_engineer.create_feature_matrix(df)

    # Get churn labels
    churn_def = ChurnDefinition(inactivity_days=30)
    labels_df = churn_def.create_churn_labels(df)

    # Merge features with labels
    model_df = features_df.merge(
        labels_df[["userId", "is_churned", "churn_type"]], on="userId", how="inner"
    ).dropna(subset=["is_churned"])

    print("=== MODEL TRAINING ===")
    print(f"Dataset shape: {model_df.shape}")
    print(f"Churn rate: {model_df['is_churned'].mean():.3f}")

    # Train models
    trainer = ChurnModelTrainer()
    results = trainer.train_all_models(model_df)

    # Print results
    print("\n=== MODEL COMPARISON ===")
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(f"\n{model_name.upper()}:")
        print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Recall@90% Precision: {metrics['recall_at_90_precision']:.3f}")

    # Error analysis
    print("\n=== ERROR ANALYSIS ===")
    error_analysis = trainer.error_analysis(model_df)
    print(f"False Positives: {error_analysis['false_positives_count']}")
    print(f"False Negatives: {error_analysis['false_negatives_count']}")

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
