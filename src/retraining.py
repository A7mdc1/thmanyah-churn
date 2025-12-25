"""
Automated retraining pipeline for churn prediction models.
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from .churn_definition import ChurnDefinition
from .data import ChurnDataLoader
from .feature_engineering import ChurnFeatureEngineer
from .model import ChurnModelTrainer

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated pipeline for model retraining based on triggers."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.last_training_time = None
        self.retrain_log_path = Path("logs/retrain_log.json")
        self.retrain_log_path.parent.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load retraining configuration."""
        # Default configuration
        default_config = {
            "retraining": {
                "schedule_days": 7,  # Retrain every 7 days
                # 10% performance drop triggers retraining
                "performance_threshold": 0.1,
                "drift_threshold": 0.1,  # Drift threshold
                "min_new_data_size": 100,  # Minimum new data points
                "max_training_days": 90,  # Use last 90 days of data
            },
            "model": {
                "target_metric": "pr_auc",
                "improvement_threshold": 0.02,  # 2% improvement to deploy new model
            },
        }

        try:
            import yaml

            if Path(config_path).exists():
                with open(config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except Exception:
            logger.warning(f"Could not load config from {config_path}, using defaults")

        return default_config

    def _log_retrain_event(self, event_type: str, details: Dict):
        """Log retraining events."""
        log_entry = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "details": details,
        }

        # Load existing log
        if self.retrain_log_path.exists():
            try:
                with open(self.retrain_log_path, "r") as f:
                    log_history = json.load(f)
            except Exception:
                log_history = []
        else:
            log_history = []

        log_history.append(log_entry)

        # Save updated log
        with open(self.retrain_log_path, "w") as f:
            json.dump(log_history, f, default=str, indent=2)

        logger.info(f"Logged retraining event: {event_type}")

    def get_incremental_data(self, full_data: pd.DataFrame) -> pd.DataFrame:
        """Get new data since last training based on DATA timestamps."""
        # Load last data timestamp from model_info.json
        model_info_path = Path("models/model_info.json")
        last_data_ts = None

        if model_info_path.exists():
            try:
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                last_data_ts = model_info.get("last_data_ts")
                if last_data_ts:
                    last_data_ts = pd.to_datetime(last_data_ts)
            except Exception:
                pass

        if last_data_ts is None:
            # First training - use all data from last 7 days
            cutoff_time = pd.to_datetime(full_data["ts"]).max() - timedelta(days=7)
            new_data = full_data[pd.to_datetime(full_data["ts"]) > cutoff_time]
            logger.info("First training: using last 7 days of data")
        else:
            # Get data newer than last training DATA timestamp
            new_data = full_data[pd.to_datetime(full_data["ts"]) > last_data_ts]
            logger.info(f"Incremental data since {last_data_ts}")

        logger.info(
            f"New data: {len(new_data)} events, "
            f"{new_data['userId'].nunique()} unique users"
        )
        return new_data

    def check_retraining_triggers(
        self, new_data: pd.DataFrame, current_performance: Dict
    ) -> Dict:
        """Check if retraining should be triggered."""
        triggers = {
            "schedule_trigger": False,
            "performance_trigger": False,
            "drift_trigger": False,
            "data_volume_trigger": False,
            "should_retrain": False,
            "reasons": [],
            "new_events": len(new_data),
            "new_users": new_data["userId"].nunique() if len(new_data) > 0 else 0,
        }

        # 1. Schedule-based trigger
        schedule_days = self.config["retraining"]["schedule_days"]
        model_info_path = Path("models/model_info.json")

        if model_info_path.exists():
            try:
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                last_training_time = pd.to_datetime(
                    model_info.get("last_training_time")
                )
                days_since_training = (datetime.now() - last_training_time).days

                if days_since_training >= schedule_days:
                    triggers["schedule_trigger"] = True
                    triggers["reasons"].append(
                        f"Scheduled retraining ({days_since_training} days)"
                    )
            except Exception:
                triggers["schedule_trigger"] = True
                triggers["reasons"].append("No previous training found")
        else:
            # First training - always trigger
            triggers["schedule_trigger"] = True
            triggers["reasons"].append("First training - no model exists")

        # 2. Performance degradation trigger
        perf_threshold = self.config["retraining"]["performance_threshold"]
        target_metric = self.config["model"]["target_metric"]

        if target_metric in current_performance and model_info_path.exists():
            try:
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                baseline_performance = model_info["performance_metrics"].get(
                    target_metric, 0.9
                )
                performance_drop = (
                    baseline_performance - current_performance[target_metric]
                )

                if performance_drop > perf_threshold:
                    triggers["performance_trigger"] = True
                    triggers["reasons"].append(
                        f"Performance drop: {performance_drop:.3f}"
                    )
            except Exception:
                pass

        # 3. Data drift trigger using PSI on key features
        if len(new_data) > 0:
            drift_detected = self._detect_drift(new_data)
            if drift_detected:
                triggers["drift_trigger"] = True
                triggers["reasons"].append("Data drift detected via PSI")

        # 4. Data volume trigger
        min_data_size = self.config["retraining"]["min_new_data_size"]
        if len(new_data) >= min_data_size:
            triggers["data_volume_trigger"] = True
            triggers["reasons"].append(f"Sufficient new data: {len(new_data)} events")
        else:
            triggers["reasons"].append(
                f"Insufficient data: {len(new_data)} < {min_data_size}"
            )

        # Decision logic: retrain if any trigger is active AND we have enough data
        triggers["should_retrain"] = (
            triggers["schedule_trigger"]
            or triggers["performance_trigger"]
            or triggers["drift_trigger"]
        ) and triggers["data_volume_trigger"]

        return triggers

    def _detect_drift(self, new_data: pd.DataFrame) -> bool:
        """Detect data drift using PSI on engineered features."""
        try:
            # Load reference data from previous training period
            model_info_path = Path("models/model_info.json")
            if not model_info_path.exists():
                logger.info("No previous model info found, skipping drift detection")
                return False

            # Get reference data window (data used in previous training)
            with open(model_info_path, "r") as f:
                model_info = json.load(f)

            last_data_ts = model_info.get("last_data_ts")
            if not last_data_ts:
                logger.info("No last_data_ts found, skipping drift detection")
                return False

            # Load full data and extract reference window
            from .config import DATA_DIR
            loader = ChurnDataLoader(data_path=DATA_DIR)
            full_data = loader.load_event_logs("customer_churn_mini.json")
            full_data = loader.basic_preprocessing(full_data)

            # Reference window: data used in previous training
            last_data_ts_dt = pd.to_datetime(last_data_ts)
            reference_window = full_data[
                pd.to_datetime(full_data["ts"]) <= last_data_ts_dt
            ]

            # Build engineered feature matrices
            from .feature_engineering import ChurnFeatureEngineer

            feature_engineer = ChurnFeatureEngineer()

            # Reference features (from previous training data)
            ref_features = feature_engineer.create_feature_matrix(reference_window)

            # Current features (from new data)
            if len(new_data) == 0:
                logger.info("No new data for drift detection")
                return False

            curr_features = feature_engineer.create_feature_matrix(new_data)

            # Select key numeric features for drift detection
            numeric_cols = ref_features.select_dtypes(include=[np.number]).columns
            key_features = [
                col
                for col in numeric_cols
                if col != "userId" and col in curr_features.columns
            ][
                :10
            ]  # Top 10

            drift_threshold = self.config["retraining"]["drift_threshold"]
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            drift_scores = {}
            for feature in key_features:
                if feature in ref_features.columns and feature in curr_features.columns:
                    ref_values = ref_features[feature].dropna()
                    curr_values = curr_features[feature].dropna()

                    if len(ref_values) > 0 and len(curr_values) > 0:
                        psi_score = self._calculate_psi(ref_values, curr_values)
                        drift_scores[feature] = psi_score

            # Save drift report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            drift_report = {
                "timestamp": datetime.now().isoformat(),
                "drift_scores": drift_scores,
                "threshold": drift_threshold,
                "drift_detected": any(
                    score > drift_threshold for score in drift_scores.values()
                ),
            }

            with open(reports_dir / f"drift_report_{timestamp}.json", "w") as f:
                json.dump(drift_report, f, indent=2)

            max_drift = max(drift_scores.values()) if drift_scores else 0
            logger.info(
                f"Drift check: max PSI = {max_drift:.3f}, threshold = {drift_threshold}"
            )

            return max_drift > drift_threshold

        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
            return False

    def _calculate_psi(
        self, reference: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference distribution
            _, bin_edges = np.histogram(reference, bins=bins)

            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)

            # Convert to proportions (avoid division by zero)
            ref_props = (ref_counts + 1e-6) / (ref_counts.sum() + bins * 1e-6)
            cur_props = (cur_counts + 1e-6) / (cur_counts.sum() + bins * 1e-6)

            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            return abs(psi)

        except Exception:
            return 0.0

    def prepare_training_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for retraining."""
        # Use all available data for retraining demo
        logger.info(f"Using {len(new_data)} records for retraining")
        return new_data

    def retrain_model(self, training_data: pd.DataFrame) -> Dict:
        """Retrain the model with new data."""
        try:
            logger.info("Starting model retraining")

            # Set up MLflow tracking
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("thmanyah-churn")

            # Feature engineering
            feature_engineer = ChurnFeatureEngineer()
            features_df = feature_engineer.create_feature_matrix(training_data)

            # Get churn labels
            churn_def = ChurnDefinition()
            labels_df = churn_def.create_churn_labels(training_data)

            # Merge features with labels
            model_df = features_df.merge(
                labels_df[["userId", "is_churned", "churn_type"]],
                on="userId",
                how="inner",
            ).dropna(subset=["is_churned"])

            logger.info(f"Training dataset shape: {model_df.shape}")

            # Validate perfect metrics with time-forward evaluation
            perfect_metric_threshold = 0.99
            if len(model_df) > 50:  # Only validate if sufficient data
                temporal_val_results = self._validate_temporal_performance(model_df)
                if temporal_val_results["avg_pr_auc"] >= perfect_metric_threshold:
                    logger.warning(
                        f"Perfect/near-perfect metrics detected "
                        f"(PR-AUC: {temporal_val_results['avg_pr_auc']:.4f})"
                    )
                    logger.warning(
                        "This may indicate data leakage or overfitting. Investigate:"
                    )
                    logger.warning("1. Check temporal cutoffs in feature engineering")
                    logger.warning("2. Verify no future information in features")
                    logger.warning("3. Consider reducing model complexity")

            # Train new model with MLflow tracking
            with mlflow.start_run(run_name="retraining"):
                # Log retraining parameters
                mlflow.log_param("training_type", "retraining")
                mlflow.log_param("training_data_size", len(model_df))
                mlflow.log_param(
                    "data_timestamp",
                    pd.to_datetime(training_data["ts"]).max().isoformat(),
                )

                trainer = ChurnModelTrainer()
                results = trainer.train_all_models(model_df)

                # Get best model performance and log additional metrics
                best_model_name = max(
                    results.keys(), key=lambda k: results[k]["metrics"]["pr_auc"]
                )
                best_performance = results[best_model_name]["metrics"]
                mlflow.log_metric("best_pr_auc", best_performance["pr_auc"])
                mlflow.log_metric("churn_rate", model_df["is_churned"].mean())

                # Log model metadata as artifact
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                max_data_ts = pd.to_datetime(training_data["ts"]).max().isoformat()
                model_info_data = {
                    "baseline_pr_auc": best_performance["pr_auc"],
                    "last_training_time": datetime.now().isoformat(),
                    "last_data_ts": max_data_ts,
                    "best_model_type": best_model_name,
                    "model_file": f"model_{timestamp}.pkl",
                    "training_data_size": len(model_df),
                    "performance_metrics": best_performance,
                }

                # Save and log model_info as artifact
                model_info_path = f"model_info_{timestamp}.json"
                with open(model_info_path, "w") as f:
                    json.dump(model_info_data, f, indent=2)
                mlflow.log_artifact(model_info_path)

                # Log drift report if it exists
                reports_dir = Path("reports")
                latest_drift_report = list(reports_dir.glob("drift_report_*.json"))
                if latest_drift_report:
                    latest_drift_report.sort(
                        key=lambda x: x.stat().st_mtime, reverse=True
                    )
                    mlflow.log_artifact(str(latest_drift_report[0]))

            # Get best model performance
            best_model_name = max(
                results.keys(), key=lambda k: results[k]["metrics"]["pr_auc"]
            )
            best_performance = results[best_model_name]["metrics"]
            best_model = results[best_model_name]["model"]

            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Save timestamped model
            timestamped_model_path = models_dir / f"model_{timestamp}.pkl"
            with open(timestamped_model_path, "wb") as f:
                pickle.dump(best_model, f)

            logger.info(f"Model saved as: {timestamped_model_path}")

            # Load baseline performance from existing model_info.json
            model_info_path = models_dir / "model_info.json"
            baseline_performance = None

            if model_info_path.exists():
                try:
                    with open(model_info_path, "r") as f:
                        existing_info = json.load(f)
                    baseline_performance = existing_info["performance_metrics"].get(
                        "pr_auc"
                    )
                    logger.info(f"Loaded baseline PR-AUC: {baseline_performance}")
                except Exception:
                    logger.warning("Could not load baseline performance")

            # Calculate improvement and determine deployment
            improvement_threshold = self.config["model"]["improvement_threshold"]
            should_deploy = False
            performance_improvement = 0.0

            if baseline_performance is None:
                # First model - always deploy
                should_deploy = True
                logger.info("First model - deploying automatically")
            else:
                performance_improvement = (
                    best_performance["pr_auc"] - baseline_performance
                )
                should_deploy = performance_improvement >= improvement_threshold
                logger.info(
                    f"Performance improvement: {performance_improvement:.4f} "
                    f"(threshold: {improvement_threshold})"
                )

            # Only update model_info.json if deploying
            if should_deploy:
                # Get max timestamp from training data for last_data_ts
                max_data_ts = pd.to_datetime(training_data["ts"]).max().isoformat()

                # Get feature list from trained model
                feature_list = list(
                    results[best_model_name]["model"]
                    .named_steps["scaler"]
                    .feature_names_in_
                )

                model_info = {
                    "baseline_pr_auc": best_performance["pr_auc"],
                    "last_training_time": datetime.now().isoformat(),
                    "last_data_ts": max_data_ts,
                    "best_model_type": best_model_name,
                    "model_file": f"model_{timestamp}.pkl",
                    "training_data_size": len(model_df),
                    "performance_metrics": best_performance,
                    "feature_list": feature_list,
                }

                with open(model_info_path, "w") as f:
                    json.dump(model_info, f, indent=2)

                # Update latest_model.txt to point to new model
                latest_model_path = models_dir / "latest_model.txt"
                with open(latest_model_path, "w") as f:
                    f.write(f"model_{timestamp}.pkl")

                logger.info("Updated model_info.json with new baseline")
                logger.info(
                    f"Updated latest_model.txt to point to: model_{timestamp}.pkl"
                )
            else:
                logger.info("Model not deployed - insufficient improvement")

            retraining_result = {
                "success": True,
                "timestamp": datetime.now(),
                "best_model": best_model_name,
                "performance": best_performance,
                "performance_improvement": performance_improvement,
                "should_deploy": should_deploy,
                "training_data_size": len(model_df),
                "churn_rate": model_df["is_churned"].mean(),
                "model_file": f"model_{timestamp}.pkl",
            }

            self.last_training_time = datetime.now()

            logger.info(
                f"Retraining complete. Best model: {best_model_name}, "
                f"PR-AUC: {best_performance['pr_auc']:.3f}"
            )
            logger.info(f"Model saved as: {timestamped_model_path}")

            return retraining_result

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {"success": False, "timestamp": datetime.now(), "error": str(e)}

    def run_retraining_check(
        self, new_data: pd.DataFrame, current_performance: Dict = None
    ) -> Dict:
        """Main function to check and execute retraining if needed."""
        logger.info("Running retraining check")

        if current_performance is None:
            current_performance = {"pr_auc": 0.8}  # Default

        # Check triggers
        triggers = self.check_retraining_triggers(new_data, current_performance)

        result = {
            "timestamp": datetime.now(),
            "triggers": triggers,
            "retraining_executed": False,
        }

        if triggers["should_retrain"]:
            logger.info(f"Retraining triggered: {', '.join(triggers['reasons'])}")

            # Prepare data and retrain
            training_data = self.prepare_training_data(new_data)
            retrain_result = self.retrain_model(training_data)

            result["retraining_executed"] = True
            result["retraining_result"] = retrain_result

            # Log event
            self._log_retrain_event(
                "retraining_executed", {"triggers": triggers, "result": retrain_result}
            )

        else:
            logger.info("No retraining triggers activated")
            self._log_retrain_event(
                "retraining_check",
                {"triggers": triggers, "action": "no_retraining_needed"},
            )

        return result

    def _validate_temporal_performance(self, model_df: pd.DataFrame) -> Dict:
        """
        Validate model performance using strict time-forward evaluation to
        detect leakage.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import auc, precision_recall_curve
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        logger.info("Running temporal validation to detect potential data leakage")

        # Prepare features and target
        feature_cols = [
            col
            for col in model_df.columns
            if col not in ["userId", "is_churned", "churn_type"]
        ]
        X = model_df[feature_cols].fillna(0)
        y = model_df["is_churned"]

        # Use time series split for temporal validation
        tscv = TimeSeriesSplit(n_splits=3)
        pr_aucs = []

        try:
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Quick logistic regression for validation
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                class_weight="balanced", random_state=42
                            ),
                        ),
                    ]
                )

                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)[:, 1]

                # Calculate PR-AUC
                if len(set(y_test)) > 1:  # Only if both classes present
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    pr_auc = auc(recall, precision)
                    pr_aucs.append(pr_auc)

        except Exception as e:
            logger.warning(f"Temporal validation failed: {e}")
            return {"avg_pr_auc": 0.0, "splits": 0}

        avg_pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0
        logger.info(
            f"Temporal validation: {len(pr_aucs)} splits, avg PR-AUC: {avg_pr_auc:.4f}"
        )

        return {
            "avg_pr_auc": avg_pr_auc,
            "splits": len(pr_aucs),
            "individual_scores": pr_aucs,
        }


class RetrainingScheduler:
    """Scheduler for automated retraining checks."""

    def __init__(self, pipeline: RetrainingPipeline):
        self.pipeline = pipeline
        self.is_running = False

    def start_scheduler(self, check_interval_hours: int = 24):
        """Start the retraining scheduler (simplified version)."""
        logger.info(
            f"Retraining scheduler started (check every {check_interval_hours} hours)"
        )

        # In a real implementation, this would use a proper scheduler like
        # Celery or APScheduler
        # For demo purposes, we'll just log the configuration

        schedule_config = {
            "scheduler_type": "periodic",
            "check_interval_hours": check_interval_hours,
            "config": self.pipeline.config,
            "status": "configured",
        }

        return schedule_config


def main():
    """End-to-end retraining pipeline on real dataset."""
    print("=== END-TO-END RETRAINING ON REAL DATASET ===")

    # Initialize pipeline
    pipeline = RetrainingPipeline()

    # Load real data using ChurnDataLoader
    from .config import DATA_DIR
    loader = ChurnDataLoader(data_path=DATA_DIR)
    print("Loading customer_churn_mini.json...")
    full_data = loader.load_event_logs("customer_churn_mini.json")
    full_data = loader.basic_preprocessing(full_data)

    # Get incremental data since last training
    new_data = pipeline.get_incremental_data(full_data)

    print(
        f"Full dataset: {len(full_data)} events, {full_data['userId'].nunique()} users"
    )
    print(
        f"Incremental data: {len(new_data)} events, "
        f"{new_data['userId'].nunique()} users"
    )

    # Set current performance for retraining check
    # For first run with no model, this won't matter since schedule trigger will fire
    current_performance = {
        "pr_auc": 0.95,  # High performance to test other triggers
        "accuracy": 0.92,
    }

    # Run retraining check
    result = pipeline.run_retraining_check(new_data, current_performance)

    print("\n=== RETRAINING RESULTS ===")
    print(f"Should retrain: {result['triggers']['should_retrain']}")
    print(f"Reasons: {result['triggers']['reasons']}")
    print(f"Retraining executed: {result['retraining_executed']}")

    if result["retraining_executed"] and "retraining_result" in result:
        retrain_result = result["retraining_result"]
        if retrain_result["success"]:
            print("\n=== MODEL TRAINING SUCCESS ===")
            print(f"Best model: {retrain_result['best_model']}")
            print(f"PR-AUC: {retrain_result['performance']['pr_auc']:.3f}")
            print(f"Training data size: {retrain_result['training_data_size']}")
            print(f"Model saved as: {retrain_result['model_file']}")
        else:
            print("\n=== MODEL TRAINING FAILED ===")
            print(f"Error: {retrain_result['error']}")

    return result


if __name__ == "__main__":
    main()
