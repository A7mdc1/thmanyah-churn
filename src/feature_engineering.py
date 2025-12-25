"""
Feature engineering for customer churn prediction.

This module creates features from event logs while preventing data leakage
by using only information available at the time of prediction.
"""

import logging
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    """
    Feature engineering for churn prediction with data leakage prevention.

    All features use only historical data available at prediction time.
    """

    def __init__(self, prediction_cutoff: Optional[str] = None):
        """
        Initialize feature engineer.

        Args:
            prediction_cutoff: ISO date string for feature cutoff
                (prevents data leakage)
        """
        self.prediction_cutoff = (
            pd.to_datetime(prediction_cutoff) if prediction_cutoff else None
        )

    def apply_temporal_cutoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal cutoff to prevent data leakage."""
        if self.prediction_cutoff:
            return df[df["ts"] <= self.prediction_cutoff].copy()
        return df.copy()

    def extract_temporal_features(self, df: pd.DataFrame, user_id: str) -> Dict:
        """
        Extract temporal features for a single user.

        Args:
            df: User's event data
            user_id: User identifier

        Returns:
            Dictionary of temporal features
        """
        if len(df) == 0:
            return {}

        df_sorted = df.sort_values("ts")
        first_event = df_sorted["ts"].min()
        last_event = df_sorted["ts"].max()
        registration = df_sorted["registration"].iloc[0]

        # Calculate cutoff date for features (prevent data leakage)
        cutoff_date = self.prediction_cutoff or df["ts"].max()

        features = {
            # Basic temporal features
            "days_since_registration": (cutoff_date - registration).days,
            "days_since_first_event": (cutoff_date - first_event).days,
            "days_since_last_event": (cutoff_date - last_event).days,
            "user_activity_span_days": (last_event - first_event).days,
            # Activity frequency
            "total_events": len(df),
            "events_per_day": len(df) / max((last_event - first_event).days, 1),
            # Session patterns
            "unique_sessions": df["sessionId"].nunique(),
            "avg_events_per_session": len(df) / max(df["sessionId"].nunique(), 1),
        }

        # Recent activity patterns (last 7, 14, 30 days)
        for days in [7, 14, 30]:
            recent_cutoff = cutoff_date - timedelta(days=days)
            recent_events = df[df["ts"] >= recent_cutoff]

            features.update(
                {
                    f"events_last_{days}d": len(recent_events),
                    f"sessions_last_{days}d": recent_events["sessionId"].nunique(),
                    f"days_active_last_{days}d": recent_events["ts"].dt.date.nunique(),
                    f"avg_events_per_day_last_{days}d": len(recent_events) / days,
                }
            )

        return features

    def extract_behavioral_features(self, df: pd.DataFrame, user_id: str) -> Dict:
        """
        Extract behavioral engagement features.

        Args:
            df: User's event data
            user_id: User identifier

        Returns:
            Dictionary of behavioral features
        """
        if len(df) == 0:
            return {}

        features = {}

        # Event type distribution
        event_counts = df["page"].value_counts()
        total_events = len(df)

        # Core engagement events
        features.update(
            {
                "songs_played": event_counts.get("NextSong", 0),
                "thumbs_up": event_counts.get("Thumbs Up", 0),
                "thumbs_down": event_counts.get("Thumbs Down", 0),
                "playlist_additions": event_counts.get("Add to Playlist", 0),
                "friend_additions": event_counts.get("Add Friend", 0),
                "roll_adverts": event_counts.get("Roll Advert", 0),
            }
        )

        # Ratios and engagement metrics
        if total_events > 0:
            features.update(
                {
                    "songs_ratio": features["songs_played"] / total_events,
                    "engagement_ratio": (
                        features["thumbs_up"] + features["thumbs_down"]
                    )
                    / max(features["songs_played"], 1),
                    "positive_feedback_ratio": features["thumbs_up"]
                    / max(features["thumbs_up"] + features["thumbs_down"], 1),
                    "playlist_ratio": features["playlist_additions"]
                    / max(features["songs_played"], 1),
                    "social_ratio": features["friend_additions"] / total_events,
                }
            )

        # Music listening patterns
        song_events = df[df["page"] == "NextSong"]
        if len(song_events) > 0:
            features.update(
                {
                    "unique_artists": song_events["artist"].nunique(),
                    "unique_songs": song_events["song"].nunique(),
                    "avg_song_length": song_events["length"].mean(),
                    "total_listening_time": song_events["length"].sum(),
                    "artist_diversity": song_events["artist"].nunique()
                    / len(song_events),
                    "song_diversity": song_events["song"].nunique() / len(song_events),
                }
            )

        # Session behavior
        session_stats = (
            df.groupby("sessionId")
            .agg(
                {
                    "ts": ["count", "min", "max"],
                    "page": lambda x: (x == "NextSong").sum(),
                }
            )
            .reset_index()
        )

        session_stats.columns = [
            "sessionId",
            "events_per_session",
            "session_start",
            "session_end",
            "songs_per_session",
        ]
        session_stats["session_duration_minutes"] = (
            session_stats["session_end"] - session_stats["session_start"]
        ).dt.total_seconds() / 60

        if len(session_stats) > 0:
            features.update(
                {
                    "avg_session_duration": session_stats[
                        "session_duration_minutes"
                    ].mean(),
                    "avg_songs_per_session": session_stats["songs_per_session"].mean(),
                    "max_session_duration": session_stats[
                        "session_duration_minutes"
                    ].max(),
                    "session_duration_std": session_stats[
                        "session_duration_minutes"
                    ].std()
                    or 0,
                }
            )

        return features

    def extract_subscription_features(self, df: pd.DataFrame, user_id: str) -> Dict:
        """
        Extract subscription and payment-related features.

        Args:
            df: User's event data
            user_id: User identifier

        Returns:
            Dictionary of subscription features
        """
        if len(df) == 0:
            return {}

        features = {}

        # Current and historical subscription levels
        level_history = df.drop_duplicates(subset=["level"], keep="first").sort_values(
            "ts"
        )
        current_level = df["level"].iloc[-1]

        features.update(
            {
                "current_level_paid": int(current_level == "paid"),
                "current_level_free": int(current_level == "free"),
                "subscription_changes": len(level_history) - 1,
                "ever_paid": int("paid" in df["level"].values),
                "ever_free": int("free" in df["level"].values),
            }
        )

        # Subscription change patterns
        if len(level_history) > 1:
            level_sequence = level_history["level"].tolist()
            features.update(
                {
                    "upgraded_to_paid": int(
                        "free" in level_sequence
                        and "paid" in level_sequence
                        and level_sequence.index("paid") > level_sequence.index("free")
                    ),
                    "downgraded_to_free": int(
                        "paid" in level_sequence
                        and "free" in level_sequence
                        and level_sequence.index("free") > level_sequence.index("paid")
                    ),
                }
            )

        # Time in each subscription level
        paid_events = df[df["level"] == "paid"]
        free_events = df[df["level"] == "free"]

        features.update(
            {
                "days_as_paid": (
                    (paid_events["ts"].max() - paid_events["ts"].min()).days
                    if len(paid_events) > 0
                    else 0
                ),
                "days_as_free": (
                    (free_events["ts"].max() - free_events["ts"].min()).days
                    if len(free_events) > 0
                    else 0
                ),
                "events_as_paid": len(paid_events),
                "events_as_free": len(free_events),
            }
        )

        # Downgrade events (strong churn signal)
        downgrade_events = df[df["page"].str.contains("Downgrade", na=False)]
        features.update(
            {
                "downgrade_events": len(downgrade_events),
                "days_since_downgrade": (
                    (df["ts"].max() - downgrade_events["ts"].max()).days
                    if len(downgrade_events) > 0
                    else 999
                ),
            }
        )

        return features

    def extract_trend_features(
        self, df: pd.DataFrame, user_id: str, window_days: int = 7
    ) -> Dict:
        """
        Extract trend features showing user behavior changes.

        Args:
            df: User's event data
            user_id: User identifier
            window_days: Window size for trend calculation

        Returns:
            Dictionary of trend features
        """
        if len(df) == 0:
            return {}

        features = {}
        cutoff_date = self.prediction_cutoff or df["ts"].max()

        # Calculate rolling metrics
        df_sorted = df.sort_values("ts")
        df_sorted["date"] = df_sorted["ts"].dt.date

        daily_stats = (
            df_sorted.groupby("date")
            .agg(
                {
                    "ts": "count",
                    "sessionId": "nunique",
                    "page": lambda x: (x == "NextSong").sum(),
                }
            )
            .rename(
                columns={
                    "ts": "daily_events",
                    "sessionId": "daily_sessions",
                    "page": "daily_songs",
                }
            )
        )

        if len(daily_stats) >= window_days:
            # Recent vs historical comparison
            recent_days = daily_stats.tail(window_days)
            historical_days = (
                daily_stats.head(-window_days)
                if len(daily_stats) > window_days
                else daily_stats
            )

            if len(historical_days) > 0:
                features.update(
                    {
                        "activity_trend_events": recent_days["daily_events"].mean()
                        / max(historical_days["daily_events"].mean(), 1),
                        "activity_trend_sessions": recent_days["daily_sessions"].mean()
                        / max(historical_days["daily_sessions"].mean(), 1),
                        "activity_trend_songs": recent_days["daily_songs"].mean()
                        / max(historical_days["daily_songs"].mean(), 1),
                    }
                )

        # Activity decline patterns
        for days in [3, 7, 14]:
            recent_window = cutoff_date - timedelta(days=days)
            recent_activity = df[df["ts"] >= recent_window]

            total_span_days = max((df["ts"].max() - df["ts"].min()).days, 1)
            expected_recent = max(len(df) * (days / total_span_days), 1)

            features[f"activity_decline_{days}d"] = 1 - (
                len(recent_activity) / expected_recent
            )

        return features

    def extract_demographic_features(self, df: pd.DataFrame, user_id: str) -> Dict:
        """
        Extract user demographic features.

        Args:
            df: User's event data
            user_id: User identifier

        Returns:
            Dictionary of demographic features
        """
        if len(df) == 0:
            return {}

        # Static demographic info (from first record)
        first_record = df.iloc[0]

        features = {
            "gender_M": int(first_record.get("gender") == "M"),
            "gender_F": int(first_record.get("gender") == "F"),
            "gender_unknown": int(first_record.get("gender") not in ["M", "F"]),
        }

        # Location features (simplified)
        location = first_record.get("location", "")
        if location:
            # Extract state/region from location string
            location_parts = location.split(",")
            if len(location_parts) > 1:
                state = location_parts[-1].strip()
                features["location_state"] = state

                # Major metropolitan areas
                major_metros = ["CA", "NY", "TX", "FL", "IL"]
                features["major_metro"] = int(state in major_metros)

        return features

    def create_user_features(self, df: pd.DataFrame, user_id: str) -> Dict:
        """
        Create comprehensive feature set for a single user.

        Args:
            df: User's event data
            user_id: User identifier

        Returns:
            Dictionary of all features for the user
        """
        # Apply temporal cutoff to prevent data leakage
        df_features = self.apply_temporal_cutoff(df)

        if len(df_features) == 0:
            logger.warning(f"No data available for user {user_id} before cutoff")
            return {"userId": user_id}

        features = {"userId": user_id}

        # Extract all feature types
        features.update(self.extract_temporal_features(df_features, user_id))
        features.update(self.extract_behavioral_features(df_features, user_id))
        features.update(self.extract_subscription_features(df_features, user_id))
        features.update(self.extract_trend_features(df_features, user_id))
        features.update(self.extract_demographic_features(df_features, user_id))

        return features

    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix for all users.

        Args:
            df: Event logs DataFrame

        Returns:
            Feature matrix DataFrame
        """
        logger.info("Starting feature engineering for all users")

        all_features = []
        users = df["userId"].unique()

        for i, user_id in enumerate(users):
            if i % 50 == 0:
                logger.info(f"Processing user {i+1}/{len(users)}")

            user_data = df[df["userId"] == user_id]
            user_features = self.create_user_features(user_data, user_id)
            all_features.append(user_features)

        feature_df = pd.DataFrame(all_features)

        # Handle missing values
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_columns] = feature_df[numeric_columns].fillna(0)

        logger.info(f"Feature engineering complete. Shape: {feature_df.shape}")
        logger.info(f"Features created: {list(feature_df.columns)}")

        return feature_df


def main():
    """Demonstrate feature engineering."""
    from churn_definition import ChurnDefinition
    from data import ChurnDataLoader

    # Load data
    loader = ChurnDataLoader()
    df = loader.load_event_logs()
    df = loader.basic_preprocessing(df)

    # Create features with temporal cutoff (simulate real prediction scenario)
    cutoff_date = "2018-11-15"  # Use first part of data for features
    feature_engineer = ChurnFeatureEngineer(prediction_cutoff=cutoff_date)

    print(f"=== FEATURE ENGINEERING (Cutoff: {cutoff_date}) ===")
    features_df = feature_engineer.create_feature_matrix(df)

    # Get churn labels
    churn_def = ChurnDefinition(inactivity_days=30)
    labels_df = churn_def.create_churn_labels(df)

    # Merge features with labels
    final_df = features_df.merge(
        labels_df[["userId", "is_churned", "churn_type"]], on="userId", how="left"
    )

    print("\n=== FEATURE MATRIX SUMMARY ===")
    print(f"Shape: {final_df.shape}")
    print(f"Features: {len(final_df.columns) - 1}")  # Exclude userId
    print(f"Users: {len(final_df)}")
    print(f"Churn rate: {final_df['is_churned'].mean():.3f}")

    # Show feature importance by correlation with churn
    numeric_features = final_df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != "is_churned"]

    correlations = (
        final_df[numeric_features + ["is_churned"]]
        .corr()["is_churned"]
        .sort_values(key=abs, ascending=False)
    )

    print("\n=== TOP FEATURES (by correlation with churn) ===")
    for feature, correlation in correlations.head(15).items():
        if feature != "is_churned":
            print(f"{feature}: {correlation:.3f}")

    # Show sample of feature matrix
    print("\n=== SAMPLE FEATURES ===")
    sample_features = [
        "userId",
        "total_events",
        "songs_played",
        "days_since_last_event",
        "current_level_paid",
        "downgrade_events",
        "is_churned",
    ]
    print(final_df[sample_features].head())

    return final_df


if __name__ == "__main__":
    features_df = main()
