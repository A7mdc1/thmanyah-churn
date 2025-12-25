"""
Configuration and path constants for the project.
"""

from pathlib import Path

# Base directory is the project root
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Configuration files
CONFIG_DIR = BASE_DIR / "configs"

# MLflow tracking
MLFLOW_DIR = BASE_DIR / "mlruns"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
