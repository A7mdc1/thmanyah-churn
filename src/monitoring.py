"""
Monitoring system for data drift, concept drift, and model performance.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor for data and concept drift detection."""

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        """
        Initialize drift monitor.

        Args:
            reference_data: Reference dataset for comparison
            threshold: Drift threshold for alerting
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_history = []

    def calculate_feature_drift(self, new_data: pd.DataFrame, feature: str) -> Dict:
        """Calculate drift for a single feature."""
        if (
            feature not in self.reference_data.columns
            or feature not in new_data.columns
        ):
            return {
                "drift_score": 0,
                "drift_detected": False,
                "method": "missing_feature",
            }

        ref_values = self.reference_data[feature].dropna()
        new_values = new_data[feature].dropna()

        if len(ref_values) == 0 or len(new_values) == 0:
            return {
                "drift_score": 0,
                "drift_detected": False,
                "method": "insufficient_data",
            }

        # Use different methods for numeric vs categorical
        if pd.api.types.is_numeric_dtype(ref_values):
            drift_score = self._numeric_drift_score(ref_values, new_values)
            method = "kolmogorov_smirnov"
        else:
            drift_score = self._categorical_drift_score(ref_values, new_values)
            method = "population_stability_index"

        drift_detected = drift_score > self.threshold

        return {
            "drift_score": drift_score,
            "drift_detected": drift_detected,
            "method": method,
            "ref_mean": (
                float(ref_values.mean())
                if pd.api.types.is_numeric_dtype(ref_values)
                else None
            ),
            "new_mean": (
                float(new_values.mean())
                if pd.api.types.is_numeric_dtype(new_values)
                else None
            ),
        }

    def _numeric_drift_score(
        self, ref_values: pd.Series, new_values: pd.Series
    ) -> float:
        """Calculate drift score for numeric features using KS test."""
        from scipy import stats

        try:
            ks_stat, _ = stats.ks_2samp(ref_values, new_values)
            return float(ks_stat)
        except Exception:
            # Fallback to simple mean difference if scipy not available
            ref_std = ref_values.std() or 1
            return abs(ref_values.mean() - new_values.mean()) / ref_std

    def _categorical_drift_score(
        self, ref_values: pd.Series, new_values: pd.Series
    ) -> float:
        """Calculate drift score for categorical features using PSI."""
        ref_dist = ref_values.value_counts(normalize=True)
        new_dist = new_values.value_counts(normalize=True)

        # Align distributions
        all_categories = set(ref_dist.index) | set(new_dist.index)
        psi_score = 0

        for category in all_categories:
            ref_pct = ref_dist.get(category, 0.001)  # Small value to avoid log(0)
            new_pct = new_dist.get(category, 0.001)
            psi_score += (new_pct - ref_pct) * np.log(new_pct / ref_pct)

        return abs(psi_score)

    def detect_data_drift(self, new_data: pd.DataFrame) -> Dict:
        """Detect data drift across all features."""
        feature_drifts = {}
        numeric_features = new_data.select_dtypes(include=[np.number]).columns
        categorical_features = new_data.select_dtypes(exclude=[np.number]).columns

        # Check numeric features
        for feature in numeric_features:
            if feature in ["is_churned", "userId"]:  # Skip target and ID
                continue
            feature_drifts[feature] = self.calculate_feature_drift(new_data, feature)

        # Check categorical features
        for feature in categorical_features:
            if feature in ["userId", "churn_type"]:  # Skip ID and target-related
                continue
            feature_drifts[feature] = self.calculate_feature_drift(new_data, feature)

        # Calculate overall drift
        drift_scores = [result["drift_score"] for result in feature_drifts.values()]
        features_with_drift = [
            f for f, result in feature_drifts.items() if result["drift_detected"]
        ]

        overall_drift = {
            "timestamp": datetime.now(),
            "overall_drift_detected": len(features_with_drift) > 0,
            "num_features_with_drift": len(features_with_drift),
            "features_with_drift": features_with_drift,
            "max_drift_score": max(drift_scores) if drift_scores else 0,
            "avg_drift_score": np.mean(drift_scores) if drift_scores else 0,
            "feature_drifts": feature_drifts,
        }

        self.drift_history.append(overall_drift)
        return overall_drift

    def detect_concept_drift(
        self, new_data: pd.DataFrame, predictions: np.ndarray
    ) -> Dict:
        """Detect concept drift by monitoring model performance."""
        if "is_churned" not in new_data.columns:
            return {"concept_drift_detected": False, "reason": "no_ground_truth"}

        # Calculate performance metrics
        y_true = new_data["is_churned"].astype(int)
        y_pred = (predictions > 0.5).astype(int)

        current_accuracy = (y_true == y_pred).mean()
        current_precision = (y_true * y_pred).sum() / max(y_pred.sum(), 1)
        current_recall = (y_true * y_pred).sum() / max(y_true.sum(), 1)

        # Get reference performance (assuming stored from training)
        ref_performance = getattr(
            self,
            "reference_performance",
            {"accuracy": 0.9, "precision": 0.9, "recall": 0.85},  # Default values
        )

        # Calculate performance degradation
        accuracy_drop = ref_performance["accuracy"] - current_accuracy
        precision_drop = ref_performance["precision"] - current_precision
        recall_drop = ref_performance["recall"] - current_recall

        # Detect significant drops (> 10%)
        concept_drift_detected = (
            accuracy_drop > 0.1 or precision_drop > 0.1 or recall_drop > 0.1
        )

        return {
            "timestamp": datetime.now(),
            "concept_drift_detected": concept_drift_detected,
            "current_performance": {
                "accuracy": current_accuracy,
                "precision": current_precision,
                "recall": current_recall,
            },
            "reference_performance": ref_performance,
            "performance_drops": {
                "accuracy_drop": accuracy_drop,
                "precision_drop": precision_drop,
                "recall_drop": recall_drop,
            },
        }


class ModelPerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, performance_log_path: str = "../logs/performance_log.json"):
        self.performance_log_path = Path(performance_log_path)
        self.performance_log_path.parent.mkdir(exist_ok=True)
        self.performance_history = self._load_performance_history()

    def _load_performance_history(self) -> List[Dict]:
        """Load existing performance history."""
        if self.performance_log_path.exists():
            try:
                with open(self.performance_log_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_performance_history(self):
        """Save performance history to file."""
        with open(self.performance_log_path, "w") as f:
            json.dump(self.performance_history, f, default=str, indent=2)

    def log_performance(
        self, predictions: np.ndarray, actuals: np.ndarray, metadata: Dict = None
    ):
        """Log model performance metrics."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        y_pred_binary = (predictions > 0.5).astype(int)
        y_true = actuals.astype(int)

        # Calculate metrics
        accuracy = (y_true == y_pred_binary).mean()
        precision = (y_true * y_pred_binary).sum() / max(y_pred_binary.sum(), 1)
        recall = (y_true * y_pred_binary).sum() / max(y_true.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        performance_entry = {
            "timestamp": datetime.now(),
            "num_predictions": len(predictions),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "churn_rate": y_true.mean(),
            "prediction_rate": y_pred_binary.mean(),
            "avg_churn_probability": predictions.mean(),
            "metadata": metadata or {},
        }

        self.performance_history.append(performance_entry)
        self._save_performance_history()

        logger.info(
            f"Logged performance: Acc={accuracy:.3f}, Prec={precision:.3f}, "
            f"Rec={recall:.3f}"
        )
        return performance_entry

    def get_performance_trends(self, days: int = 30) -> Dict:
        """Get performance trends over specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_entries = [
            entry
            for entry in self.performance_history
            if datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
            > cutoff_date
        ]

        if not recent_entries:
            return {"error": "No recent performance data available"}

        # Calculate trends
        accuracies = [entry["accuracy"] for entry in recent_entries]
        precisions = [entry["precision"] for entry in recent_entries]
        recalls = [entry["recall"] for entry in recent_entries]

        return {
            "period_days": days,
            "num_evaluations": len(recent_entries),
            "accuracy_trend": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "trend": np.polyfit(range(len(accuracies)), accuracies, 1)[0],
            },
            "precision_trend": {
                "mean": np.mean(precisions),
                "std": np.std(precisions),
                "trend": np.polyfit(range(len(precisions)), precisions, 1)[0],
            },
            "recall_trend": {
                "mean": np.mean(recalls),
                "std": np.std(recalls),
                "trend": np.polyfit(range(len(recalls)), recalls, 1)[0],
            },
            "recent_entries": recent_entries[-5:],  # Last 5 entries
        }


def main():
    """Demonstrate monitoring capabilities."""
    # This would normally use real production data
    print("=== MONITORING SYSTEM DEMO ===")
    print("Monitoring system initialized successfully")
    print(
        "Features: Data drift detection, concept drift detection, performance tracking"
    )


if __name__ == "__main__":
    main()
