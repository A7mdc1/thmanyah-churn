"""
Churn definition and labeling utilities for customer churn prediction.

This module defines what constitutes customer churn based on the event log data
and provides utilities to label users as churned or not churned.
"""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class ChurnDefinition:
    """
    Defines and implements customer churn criteria for the music streaming platform.

    CHURN DEFINITION:
    A user is considered churned if they meet ANY of the following criteria:

    1. EXPLICIT CHURN: User has a "Submit Downgrade" event (explicit cancellation)
    2. IMPLICIT CHURN: User has been inactive for >30 consecutive days before
       dataset end
    3. SUBSCRIPTION DOWNGRADE: User downgrades from paid to free and then
       becomes inactive

    RATIONALE:
    - Explicit churn is the clearest signal - user actively cancels
    - 30-day inactivity threshold balances false positives with business relevance
    - Subscription downgrade followed by inactivity indicates gradual churn
    - Time-based approach prevents data leakage by only using past information
    """

    def __init__(self, inactivity_days: int = 30):
        """
        Initialize churn definition with parameters.

        Args:
            inactivity_days: Number of days of inactivity to consider as churn
        """
        self.inactivity_days = inactivity_days
        self.churn_definition = {
            "explicit_events": ["Submit Downgrade", "Downgrade"],
            "inactivity_threshold": inactivity_days,
            "subscription_change": "paid_to_free",
        }

    def identify_explicit_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify users with explicit churn signals (downgrade events).

        Args:
            df: Event logs DataFrame

        Returns:
            DataFrame with explicit churn users and timestamps
        """
        # Find explicit churn events
        explicit_events = df[
            df["page"].isin(self.churn_definition["explicit_events"])
        ].copy()

        # Get first churn event per user
        user_churn = (
            explicit_events.groupby("userId")
            .agg({"ts": "min", "page": "first"})
            .reset_index()
        )

        user_churn.columns = ["userId", "churn_timestamp", "churn_event"]
        user_churn["churn_type"] = "explicit"

        logger.info(f"Found {len(user_churn)} users with explicit churn signals")

        return user_churn

    def identify_subscription_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify users who downgraded subscription and became inactive.

        Args:
            df: Event logs DataFrame

        Returns:
            DataFrame with subscription churn users
        """
        subscription_churn = []

        for user_id in df["userId"].unique():
            user_data = df[df["userId"] == user_id].sort_values("ts")

            # Check if user ever had paid subscription
            if "paid" not in user_data["level"].values:
                continue

            # Find transition from paid to free
            level_changes = user_data.drop_duplicates(subset=["level"], keep="first")

            if len(level_changes) > 1:
                levels = level_changes["level"].tolist()
                if "paid" in levels and "free" in levels:
                    # Find when they went from paid to free
                    paid_periods = level_changes[level_changes["level"] == "paid"]
                    free_periods = level_changes[level_changes["level"] == "free"]

                    if len(paid_periods) > 0 and len(free_periods) > 0:
                        last_paid = paid_periods["ts"].max()
                        first_free_after = free_periods[free_periods["ts"] > last_paid]

                        if len(first_free_after) > 0:
                            downgrade_time = first_free_after["ts"].min()

                            # Check activity after downgrade
                            post_downgrade = user_data[user_data["ts"] > downgrade_time]
                            if len(post_downgrade) > 0:
                                last_activity = post_downgrade["ts"].max()
                                dataset_end = df["ts"].max()

                                days_inactive = (dataset_end - last_activity).days

                                if days_inactive >= self.inactivity_days:
                                    subscription_churn.append(
                                        {
                                            "userId": user_id,
                                            "churn_timestamp": downgrade_time,
                                            "churn_type": "subscription_downgrade",
                                            "days_inactive": days_inactive,
                                        }
                                    )

        subscription_churn_df = pd.DataFrame(subscription_churn)

        logger.info(f"Found {len(subscription_churn_df)} users with subscription churn")

        return subscription_churn_df

    def identify_inactivity_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify users with inactivity-based churn.

        Args:
            df: Event logs DataFrame

        Returns:
            DataFrame with inactivity churn users
        """
        dataset_end = df["ts"].max()

        # Get last activity per user
        user_last_activity = df.groupby("userId")["ts"].max().reset_index()
        user_last_activity.columns = ["userId", "last_activity"]

        # Calculate days since last activity
        user_last_activity["days_inactive"] = (
            dataset_end - user_last_activity["last_activity"]
        ).dt.days

        # Filter users inactive beyond threshold
        inactive_users = user_last_activity[
            user_last_activity["days_inactive"] >= self.inactivity_days
        ].copy()

        inactive_users["churn_timestamp"] = inactive_users["last_activity"]
        inactive_users["churn_type"] = "inactivity"

        logger.info(f"Found {len(inactive_users)} users with inactivity churn")

        return inactive_users[
            ["userId", "churn_timestamp", "churn_type", "days_inactive"]
        ]

    def create_churn_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive churn labels for all users.

        Args:
            df: Event logs DataFrame

        Returns:
            DataFrame with user churn labels and metadata
        """
        # Get all unique users
        all_users = pd.DataFrame({"userId": df["userId"].unique()})

        # Identify different types of churn
        explicit_churn = self.identify_explicit_churn(df)
        subscription_churn = self.identify_subscription_churn(df)
        inactivity_churn = self.identify_inactivity_churn(df)

        # Combine churn types (explicit takes priority)
        churned_users = []

        # Add explicit churn (highest priority)
        if len(explicit_churn) > 0:
            churned_users.append(explicit_churn)

        # Add subscription churn for users not already churned
        if len(subscription_churn) > 0:
            existing_churned = set()
            if len(explicit_churn) > 0:
                existing_churned.update(explicit_churn["userId"])

            new_subscription_churn = subscription_churn[
                ~subscription_churn["userId"].isin(existing_churned)
            ]
            if len(new_subscription_churn) > 0:
                churned_users.append(
                    new_subscription_churn[["userId", "churn_timestamp", "churn_type"]]
                )

        # Add inactivity churn for remaining users
        if len(inactivity_churn) > 0:
            existing_churned = set()
            if churned_users:
                for df_churn in churned_users:
                    existing_churned.update(df_churn["userId"])

            new_inactivity_churn = inactivity_churn[
                ~inactivity_churn["userId"].isin(existing_churned)
            ]
            if len(new_inactivity_churn) > 0:
                churned_users.append(
                    new_inactivity_churn[["userId", "churn_timestamp", "churn_type"]]
                )

        # Combine all churned users
        if churned_users:
            all_churned = pd.concat(churned_users, ignore_index=True)
        else:
            all_churned = pd.DataFrame(
                columns=["userId", "churn_timestamp", "churn_type"]
            )

        # Create final labels
        user_labels = all_users.merge(all_churned, on="userId", how="left")

        user_labels["is_churned"] = user_labels["churn_timestamp"].notna()
        user_labels["churn_type"] = user_labels["churn_type"].fillna("not_churned")

        # Add user metadata
        user_metadata = (
            df.groupby("userId")
            .agg(
                {
                    "ts": ["min", "max", "count"],
                    "level": lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown",
                    "registration": "first",
                    "gender": "first",
                    "location": "first",
                }
            )
            .reset_index()
        )

        user_metadata.columns = [
            "userId",
            "first_activity",
            "last_activity",
            "total_events",
            "primary_level",
            "registration",
            "gender",
            "location",
        ]

        # Merge metadata
        final_labels = user_labels.merge(user_metadata, on="userId")

        # Calculate additional metrics
        final_labels["activity_span_days"] = (
            final_labels["last_activity"] - final_labels["first_activity"]
        ).dt.days

        final_labels["days_since_registration"] = (
            final_labels["last_activity"] - final_labels["registration"]
        ).dt.days

        return final_labels

    def get_churn_statistics(self, labels_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive churn statistics.

        Args:
            labels_df: DataFrame with churn labels

        Returns:
            Dictionary with churn statistics
        """
        total_users = len(labels_df)
        churned_users = labels_df["is_churned"].sum()

        stats = {
            "total_users": total_users,
            "churned_users": churned_users,
            "churn_rate": churned_users / total_users if total_users > 0 else 0,
            "churn_type_breakdown": labels_df["churn_type"].value_counts().to_dict(),
            "churn_by_level": labels_df.groupby("primary_level")["is_churned"]
            .agg(["count", "sum", "mean"])
            .to_dict(),
            "churn_by_gender": labels_df.groupby("gender")["is_churned"]
            .agg(["count", "sum", "mean"])
            .to_dict(),
        }

        return stats


def main():
    """Demonstrate churn definition and labeling."""
    from data import ChurnDataLoader

    # Load data
    loader = ChurnDataLoader()
    df = loader.load_event_logs()
    df = loader.basic_preprocessing(df)

    # Create churn definition
    churn_def = ChurnDefinition(inactivity_days=30)

    # Generate labels
    print("=== GENERATING CHURN LABELS ===")
    labels = churn_def.create_churn_labels(df)

    # Get statistics
    stats = churn_def.get_churn_statistics(labels)

    print("\n=== CHURN ANALYSIS RESULTS ===")
    print(f"Total users: {stats['total_users']}")
    print(f"Churned users: {stats['churned_users']}")
    print(f"Overall churn rate: {stats['churn_rate']:.3f}")

    print("\n=== CHURN TYPE BREAKDOWN ===")
    for churn_type, count in stats["churn_type_breakdown"].items():
        print(f"{churn_type}: {count} ({count/stats['total_users']:.3f})")

    print("\n=== CHURN BY SUBSCRIPTION LEVEL ===")
    for level in ["paid", "free"]:
        if level in stats["churn_by_level"]["count"]:
            total = stats["churn_by_level"]["count"][level]
            churned = stats["churn_by_level"]["sum"][level]
            rate = stats["churn_by_level"]["mean"][level]
            print(f"{level}: {churned}/{total} ({rate:.3f})")

    # Show sample of labels
    print("\n=== SAMPLE CHURN LABELS ===")
    print(
        labels[
            ["userId", "is_churned", "churn_type", "primary_level", "total_events"]
        ].head(10)
    )

    return labels, stats


if __name__ == "__main__":
    labels, stats = main()
