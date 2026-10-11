"""
Centralised path & hyper-parameter configuration for the Heart Disease task.

All other modules import from here – never hard-code paths elsewhere.
"""
from pathlib import Path

# ── Project root (two levels up from this file: src/heart_disease/config.py) ──
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

# ── Data directories ──────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# ── Artefact directories ──────────────────────────────────────────────────
MODELS_DIR  = ROOT_DIR / "models"  / "heart_disease"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "heart_disease"

# ── File paths ────────────────────────────────────────────────────────────
RAW_DATA_PATH         = RAW_DIR         / "heart.csv"
ENCODED_DATA_PATH     = PROCESSED_DIR   / "heart_disease_encoded.csv"
PREDICTIONS_PATH      = PREDICTIONS_DIR / "heart_disease_predictions.csv"
MODEL_PATH            = MODELS_DIR      / "svc_model.pkl"
ENCODER_PATH          = MODELS_DIR      / "le_cp.pkl"
EDA_PLOT_PATH         = OUTPUTS_DIR     / "eda.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR     / "confusion_matrix.png"
CORRELATION_PLOT_PATH = OUTPUTS_DIR     / "correlation.png"

# ── Feature configuration ─────────────────────────────────────────────────
# Assignment constraint: exactly 4 input attributes from the UCI dataset
FEATURES = ["age", "thalach", "chol", "cp"]
TARGET   = "target"

# ── Categorical columns that need label-encoding ──────────────────────────
CATEGORICAL_FEATURES = ["cp"]

# ── Model hyper-parameters ────────────────────────────────────────────────
SVC_PARAMS = {
    "kernel":        "rbf",
    "C":             1.0,
    "gamma":         "scale",
    "random_state":  42,
}

TRAIN_TEST_SPLIT_PARAMS = {
    "test_size":    0.2,
    "random_state": 42,
}
