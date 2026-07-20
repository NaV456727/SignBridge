"""
SignBridge Configuration

All configurable values for the application should be stored here.
"""

from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_FILE = DATASET_DIR / "asl_dataset.csv"
MODEL_FILE = MODELS_DIR / "asl_sign_model.pkl"

# ==========================================================
# Camera
# ==========================================================

CAMERA_INDEX = 0
MIRROR_CAMERA = True

# ==========================================================
# Hand Detection
# ==========================================================

MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# ==========================================================
# Dataset Recording
# ==========================================================

TARGET_SAMPLES = 100

SAVE_INTERVAL = 0.15  # seconds

# ==========================================================
# Supported Gestures
# ==========================================================

GESTURES = {
    "1": "hello",
    "2": "yes",
    "3": "no",
    "4": "stop",
    "5": "love",
}

# ==========================================================
# UI
# ==========================================================

WINDOW_NAME = "SignBridge"

FONT_SCALE = 0.8
TEXT_THICKNESS = 2