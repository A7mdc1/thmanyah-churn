"""
Data loading and preprocessing utilities for customer churn prediction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from .config import DATA_DIR

logger = logging.getLogger(__name__)


class ChurnDataLoader:
    """Handles loading and initial preprocessing of churn event data."""

    def __init__(self, data_path: Union[str, Path] = DATA_DIR):
        self.data_path = Path(data_path)

    def load_event_logs(
        self, filename: str = "customer_churn_mini.json"
    ) -> pd.DataFrame:
        """
        Load event logs from JSON file and convert to DataFrame.

        Args:
            filename: Name of the JSON file containing event logs

        Returns:
            DataFrame with parsed event logs
        """
        file_path = self.data_path / filename

        logger.info(f"Loading data from {file_path}")

        # Read JSON lines
        events = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(events)
        logger.info(
            f"Loaded {len(df)} events for {df['userId'].nunique()} unique users"
        )

        return df

    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic preprocessing on event logs.

        Args:
            df: Raw event logs DataFrame

        Returns:
            Preprocessed DataFrame
        """
        df_clean = df.copy()

        # Convert timestamp to datetime
        df_clean["ts"] = pd.to_datetime(df_clean["ts"], unit="ms")
        df_clean["registration"] = pd.to_datetime(df_clean["registration"], unit="ms")

        # Sort by user and timestamp
        df_clean = df_clean.sort_values(["userId", "ts"])

        # Fill missing values for categorical columns
        categorical_cols = ["page", "auth", "method", "level", "location", "gender"]
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("Unknown")

        # Convert userId to string for consistency
        df_clean["userId"] = df_clean["userId"].astype(str)

        logger.info(f"Preprocessing complete. Shape: {df_clean.shape}")

        return df_clean

    def get_data_overview(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data overview and quality report.

        Args:
            df: Event logs DataFrame

        Returns:
            Dictionary with data overview statistics
        """
        overview = {
            "basic_stats": {
                "total_events": len(df),
                "unique_users": df["userId"].nunique(),
                "date_range": {
                    "start": df["ts"].min(),
                    "end": df["ts"].max(),
                    "duration_days": (df["ts"].max() - df["ts"].min()).days,
                },
                "unique_sessions": (
                    df["sessionId"].nunique() if "sessionId" in df.columns else None
                ),
            },
            "event_types": df["page"].value_counts().to_dict(),
            "subscription_levels": df["level"].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "user_demographics": {
                "gender_distribution": df["gender"].value_counts().to_dict(),
                "auth_status": df["auth"].value_counts().to_dict(),
            },
        }

        return overview

    def analyze_user_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level summary statistics for behavior analysis.

        Args:
            df: Event logs DataFrame

        Returns:
            DataFrame with user-level statistics
        """
        user_stats = (
            df.groupby("userId")
            .agg(
                {
                    "ts": ["min", "max", "count"],
                    "sessionId": "nunique",
                    "page": lambda x: x.value_counts().to_dict(),
                    "level": lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown",
                    "gender": "first",
                    "location": "first",
                    "registration": "first",
                }
            )
            .reset_index()
        )

        # Flatten column names
        user_stats.columns = [
            "userId",
            "first_activity",
            "last_activity",
            "total_events",
            "unique_sessions",
            "page_breakdown",
            "primary_level",
            "gender",
            "location",
            "registration_date",
        ]

        # Calculate additional metrics
        user_stats["activity_span_days"] = (
            user_stats["last_activity"] - user_stats["first_activity"]
        ).dt.days

        user_stats["days_since_registration"] = (
            user_stats["last_activity"] - user_stats["registration_date"]
        ).dt.days

        return user_stats


def main():
    """Main function to demonstrate data loading and exploration."""
    loader = ChurnDataLoader()

    # Load and preprocess data
    df = loader.load_event_logs()
    df_clean = loader.basic_preprocessing(df)

    # Get data overview
    overview = loader.get_data_overview(df_clean)

    # Print basic statistics
    print("=== DATA OVERVIEW ===")
    print(f"Total Events: {overview['basic_stats']['total_events']:,}")
    print(f"Unique Users: {overview['basic_stats']['unique_users']:,}")
    date_start = overview["basic_stats"]["date_range"]["start"]
    date_end = overview["basic_stats"]["date_range"]["end"]
    print(f"Date Range: {date_start} to {date_end}")
    print(f"Duration: {overview['basic_stats']['date_range']['duration_days']} days")

    print("\n=== EVENT TYPES ===")
    for event_type, count in list(overview["event_types"].items())[:10]:
        print(f"{event_type}: {count:,}")

    print("\n=== SUBSCRIPTION LEVELS ===")
    for level, count in overview["subscription_levels"].items():
        print(f"{level}: {count:,}")

    # User behavior analysis
    user_behavior = loader.analyze_user_behavior(df_clean)
    print("\n=== USER BEHAVIOR SUMMARY ===")
    print(f"Average events per user: {user_behavior['total_events'].mean():.1f}")
    print(
        f"Average activity span: {user_behavior['activity_span_days'].mean():.1f} days"
    )
    print(f"Users with >100 events: {(user_behavior['total_events'] > 100).sum()}")


if __name__ == "__main__":
    main()
