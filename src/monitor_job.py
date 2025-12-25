"""
Continuous monitoring job for data drift and performance tracking.
Runs independently of retraining to provide ongoing monitoring.
"""

import json
import logging
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .config import LOGS_DIR, MODELS_DIR, REPORTS_DIR
from .data import ChurnDataLoader
from .feature_engineering import ChurnFeatureEngineer

logger = logging.getLogger(__name__)


class ContinuousMonitor:
    """Standalone monitoring job for continuous drift and performance tracking."""

    def __init__(self):
        self.config = {"drift_threshold": 0.1, "performance_threshold": 0.1}

    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        try:
            # Load deployed model metadata
            model_info_path = MODELS_DIR / "model_info.json"
            if not model_info_path.exists():
                logger.error(f"No deployed model found ({model_info_path} missing)")
                return {"success": False, "error": "No deployed model"}

            with open(model_info_path, "r") as f:
                model_info = json.load(f)

            logger.info(f"Monitoring deployed model: {model_info['model_file']}")

            # Load dataset
            loader = ChurnDataLoader()
            full_data = loader.load_event_logs("customer_churn_mini.json")
            full_data = loader.basic_preprocessing(full_data)

            # Split data into reference (training) and current (monitoring) windows
            last_data_ts = pd.to_datetime(model_info.get("last_data_ts"))
            if last_data_ts is None:
                logger.warning("No last_data_ts found, using recent data split")
                # Use recent 30% as "current" data for monitoring
                cutoff_time = pd.to_datetime(full_data["ts"]).quantile(0.7)
                reference_data = full_data[
                    pd.to_datetime(full_data["ts"]) <= cutoff_time
                ]
                current_data = full_data[pd.to_datetime(full_data["ts"]) > cutoff_time]
            else:
                reference_data = full_data[
                    pd.to_datetime(full_data["ts"]) <= last_data_ts
                ]
                current_data = full_data[pd.to_datetime(full_data["ts"]) > last_data_ts]

            ref_count = len(reference_data)
            cur_count = len(current_data)
            logger.info(
                f"Reference data: {ref_count} events, Current data: {cur_count} events"
            )

            # Run data drift detection
            drift_results = self._detect_data_drift(
                reference_data, current_data, timestamp
            )

            # Run performance monitoring (concept drift proxy)
            perf_results = self._monitor_performance(
                reference_data, current_data, model_info, timestamp
            )

            # Save monitoring results
            self._save_monitoring_results(drift_results, perf_results, timestamp)

            return {
                "success": True,
                "timestamp": timestamp,
                "drift_detected": drift_results.get("drift_detected", False),
                "performance_degradation": perf_results.get(
                    "performance_degradation", False
                ),
            }

        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            return {"success": False, "error": str(e)}

    def _detect_data_drift(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame, timestamp: str
    ) -> Dict:
        """Detect data drift using PSI on engineered features."""
        if len(current_data) == 0:
            logger.info("No new data for drift detection")
            # Still create a drift report even with no new data
            drift_report = {
                "timestamp": datetime.now().isoformat(),
                "drift_scores": {},
                "threshold": self.config["drift_threshold"],
                "drift_detected": False,
                "reason": "no_new_data",
                "monitoring_type": "continuous",
            }

            # Save drift report
            with open(REPORTS_DIR / f"drift_report_{timestamp}.json", "w") as f:
                json.dump(drift_report, f, indent=2)

            return drift_report

        try:
            # Build engineered feature matrices
            feature_engineer = ChurnFeatureEngineer()

            # Reference features
            ref_features = feature_engineer.create_feature_matrix(reference_data)

            # Current features
            curr_features = feature_engineer.create_feature_matrix(current_data)

            # Select key numeric features for drift detection
            numeric_cols = ref_features.select_dtypes(include=[np.number]).columns
            key_features = [
                col
                for col in numeric_cols
                if col != "userId" and col in curr_features.columns
            ][:10]

            drift_scores = {}
            for feature in key_features:
                if feature in ref_features.columns and feature in curr_features.columns:
                    ref_values = ref_features[feature].dropna()
                    curr_values = curr_features[feature].dropna()

                    if len(ref_values) > 0 and len(curr_values) > 0:
                        psi_score = self._calculate_psi(ref_values, curr_values)
                        drift_scores[feature] = psi_score

            # Create drift report
            drift_threshold = self.config["drift_threshold"]
            drift_detected = any(
                score > drift_threshold for score in drift_scores.values()
            )

            drift_report = {
                "timestamp": datetime.now().isoformat(),
                "drift_scores": drift_scores,
                "threshold": drift_threshold,
                "drift_detected": drift_detected,
                "monitoring_type": "continuous",
            }

            # Save drift report
            with open(REPORTS_DIR / f"drift_report_{timestamp}.json", "w") as f:
                json.dump(drift_report, f, indent=2)

            max_drift = max(drift_scores.values()) if drift_scores else 0
            logger.info(
                f"Drift monitoring: max PSI = {max_drift:.3f}, "
                f"threshold = {drift_threshold}"
            )

            return drift_report

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {"drift_detected": False, "error": str(e)}

    def _monitor_performance(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        model_info: Dict,
        timestamp: str,
    ) -> Dict:
        """Monitor performance degradation (concept drift proxy)."""
        try:
            # Since we don't have real-time labels, we'll use data volume and
            # churn rate proxies
            baseline_churn_rate = model_info.get("churn_rate", 0.5)  # Fallback

            # Calculate proxy metrics
            ref_event_rate = (
                len(reference_data) / reference_data["userId"].nunique()
                if len(reference_data) > 0
                else 0
            )
            curr_event_rate = (
                len(current_data) / current_data["userId"].nunique()
                if len(current_data) > 0
                else 0
            )

            # Use event rate change as a proxy for performance degradation
            event_rate_change = (
                abs(curr_event_rate - ref_event_rate) / ref_event_rate
                if ref_event_rate > 0
                else 0
            )

            performance_threshold = self.config["performance_threshold"]
            performance_degradation = event_rate_change > performance_threshold

            perf_results = {
                "timestamp": datetime.now().isoformat(),
                "reference_event_rate": ref_event_rate,
                "current_event_rate": curr_event_rate,
                "event_rate_change": event_rate_change,
                "performance_degradation": performance_degradation,
                "threshold": performance_threshold,
                "baseline_churn_rate": baseline_churn_rate,
                "monitoring_type": "continuous",
            }

            logger.info(
                f"Performance monitoring: event rate change = {event_rate_change:.3f}, "
                f"threshold = {performance_threshold}"
            )

            return perf_results

        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return {"performance_degradation": False, "error": str(e)}

    def _calculate_psi(
        self, reference: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference distribution
            _, bin_edges = np.histogram(reference, bins=bins)

            # Calculate expected (reference) and actual (current) distributions
            expected_freq, _ = np.histogram(reference, bins=bin_edges)
            actual_freq, _ = np.histogram(current, bins=bin_edges)

            # Convert to percentages
            expected_pct = expected_freq / len(reference)
            actual_pct = actual_freq / len(current)

            # Avoid division by zero
            expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
            actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

            # Calculate PSI
            psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi = np.sum(psi_values)

            return float(psi)

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def _save_monitoring_results(
        self, drift_results: Dict, perf_results: Dict, timestamp: str
    ):
        """Save monitoring results to performance log."""
        performance_log_path = LOGS_DIR / "performance_log.json"

        # Create monitoring entry
        monitoring_entry = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_id": timestamp,
            "drift_monitoring": {
                "drift_detected": drift_results.get("drift_detected", False),
                "max_drift_score": (
                    max(drift_results.get("drift_scores", {}).values())
                    if drift_results.get("drift_scores")
                    else 0
                ),
            },
            "performance_monitoring": {
                "performance_degradation": perf_results.get(
                    "performance_degradation", False
                ),
                "event_rate_change": perf_results.get("event_rate_change", 0),
            },
        }

        # Load existing log or create new one
        if performance_log_path.exists():
            with open(performance_log_path, "r") as f:
                performance_log = json.load(f)
        else:
            performance_log = []

        # Append new entry
        performance_log.append(monitoring_entry)

        # Keep only last 100 entries
        performance_log = performance_log[-100:]

        # Save updated log
        with open(performance_log_path, "w") as f:
            json.dump(performance_log, f, indent=2)

        logger.info(f"Monitoring results saved to {performance_log_path}")


def main():
    """Run monitoring job."""
    print("=== CONTINUOUS MONITORING JOB ===")

    monitor = ContinuousMonitor()
    results = monitor.run_monitoring_cycle()

    print(f"Monitoring completed: {results}")
    return results


if __name__ == "__main__":
    main()
