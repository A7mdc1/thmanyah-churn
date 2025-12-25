"""
FastAPI application for churn prediction service.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Thmanyah Churn Prediction API",
    description="Customer churn prediction for music streaming platform",
    version="1.0.0",
)

# Global model variable
model = None
model_info = {}
feature_list = []


class UserFeatures(BaseModel):
    """User features for churn prediction."""

    model_config = ConfigDict(extra="allow")
    userId: str
    days_since_registration: float = Field(
        ..., description="Days since user registration"
    )
    total_events: float = Field(..., description="Total number of events")
    songs_played: float = Field(..., description="Number of songs played")
    current_level_paid: int = Field(..., description="1 if paid user, 0 if free")
    downgrade_events: float = Field(0, description="Number of downgrade events")
    days_since_last_event: float = Field(..., description="Days since last activity")
    engagement_ratio: float = Field(0, description="Thumbs up/down ratio to songs")
    session_frequency: float = Field(..., description="Sessions per day")

    # Optional features with defaults
    thumbs_up: float = Field(0, description="Number of thumbs up")
    thumbs_down: float = Field(0, description="Number of thumbs down")
    playlist_additions: float = Field(0, description="Number of playlist additions")
    friend_additions: float = Field(0, description="Number of friend additions")
    avg_session_duration: float = Field(
        0, description="Average session duration in minutes"
    )
    unique_artists: float = Field(0, description="Number of unique artists listened to")
    unique_songs: float = Field(0, description="Number of unique songs listened to")


class ChurnPrediction(BaseModel):
    """Churn prediction response."""

    userId: str
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    prediction_timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    users: List[UserFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[ChurnPrediction]
    summary: Dict[str, Any]


def load_model():
    """Load the latest trained model dynamically."""
    global model, model_info, feature_list

    # Check for latest_model.txt first
    latest_model_path = MODELS_DIR / "latest_model.txt"
    model_filename = None

    if latest_model_path.exists():
        try:
            with open(latest_model_path, "r") as f:
                model_filename = f.read().strip()
            # Verify the referenced file exists
            if not (MODELS_DIR / model_filename).exists():
                logger.warning(f"Model file from latest_model.txt does not exist: {model_filename}")
                model_filename = None
            else:
                logger.info(f"Loading model from latest_model.txt: {model_filename}")
        except Exception:
            logger.warning("Could not read latest_model.txt")

    # Fallback to model_info.json
    if not model_filename:
        model_info_path = MODELS_DIR / "model_info.json"
        if model_info_path.exists():
            try:
                with open(model_info_path, "r") as f:
                    info = json.load(f)
                candidate_filename = info.get("model_file")
                # Verify the referenced file exists
                if candidate_filename and (MODELS_DIR / candidate_filename).exists():
                    model_filename = candidate_filename
                    logger.info(f"Loading model from model_info.json: {model_filename}")
                elif candidate_filename:
                    logger.warning(f"Model file from model_info.json does not exist: {candidate_filename}")
            except Exception:
                logger.warning("Could not read model_info.json")

    # Final fallback to any available model
    if not model_filename:
        models_dir = MODELS_DIR
        model_files = list(models_dir.glob("*.pkl"))
        if model_files:
            # Use most recently modified .pkl file
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            model_filename = latest_model.name
            logger.warning(f"Using fallback model: {model_filename}")

    if not model_filename:
        logger.error("No model files found")
        return False

    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return False

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load feature list from model_info.json
        model_info_path = MODELS_DIR / "model_info.json"
        if model_info_path.exists():
            try:
                with open(model_info_path, "r") as f:
                    info = json.load(f)
                feature_list = info.get("feature_list", [])
                logger.info(f"Loaded feature list with {len(feature_list)} features")
            except Exception as e:
                logger.warning(f"Could not load feature list: {e}")
                feature_list = []

        model_info = {
            "model_file": model_filename,
            "model_path": str(model_path),
            "loaded_at": datetime.now(),
            "features_expected": len(feature_list),
            "version": "1.0.0",
        }

        logger.info(f"Model loaded successfully: {model_info}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def prepare_features(user_features: UserFeatures) -> pd.DataFrame:
    """Prepare user features for prediction using feature_list from model_info.json."""
    global feature_list

    if not feature_list:
        raise HTTPException(status_code=503, detail="Feature list not loaded")

    # Convert to DataFrame
    features_dict = user_features.dict()
    features_dict.pop("userId")  # Remove userId

    # Create DataFrame with single row
    df = pd.DataFrame([features_dict])

    # Add missing features with default values (0)
    for feature in feature_list:
        if feature not in df.columns:
            df[feature] = 0.0

    # Keep only expected features in exact order
    # This ignores any extra features in the request
    df = df[feature_list]

    return df


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.8:
        return "high"
    elif probability >= 0.5:
        return "medium"
    else:
        return "low"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    if not load_model():
        logger.error("Failed to load model on startup")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": model is not None,
    }


@app.get("/model/info")
async def model_info_endpoint():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model_info


@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(user_features: UserFeatures):
    """Predict churn for a single user."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        features_df = prepare_features(user_features)

        # Make prediction
        churn_prob = model.predict_proba(features_df)[0, 1]
        churn_pred = churn_prob >= 0.5
        risk_level = get_risk_level(churn_prob)

        return ChurnPrediction(
            userId=user_features.userId,
            churn_probability=float(churn_prob),
            churn_prediction=bool(churn_pred),
            risk_level=risk_level,
            prediction_timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Prediction error for user {user_features.userId}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest):
    """Predict churn for multiple users."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.users) > 1000:
        raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")

    predictions = []

    try:
        for user_features in request.users:
            # Prepare features
            features_df = prepare_features(user_features)

            # Make prediction
            churn_prob = model.predict_proba(features_df)[0, 1]
            churn_pred = churn_prob >= 0.5
            risk_level = get_risk_level(churn_prob)

            predictions.append(
                ChurnPrediction(
                    userId=user_features.userId,
                    churn_probability=float(churn_prob),
                    churn_prediction=bool(churn_pred),
                    risk_level=risk_level,
                    prediction_timestamp=datetime.now(),
                )
            )

        # Calculate summary
        churn_probs = [p.churn_probability for p in predictions]
        summary = {
            "total_users": len(predictions),
            "high_risk_users": sum(1 for p in predictions if p.risk_level == "high"),
            "medium_risk_users": sum(
                1 for p in predictions if p.risk_level == "medium"
            ),
            "low_risk_users": sum(1 for p in predictions if p.risk_level == "low"),
            "avg_churn_probability": float(np.mean(churn_probs)),
            "max_churn_probability": float(np.max(churn_probs)),
            "min_churn_probability": float(np.min(churn_probs)),
        }

        return BatchPredictionResponse(predictions=predictions, summary=summary)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Thmanyah Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
