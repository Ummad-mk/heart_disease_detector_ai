"""
Centralised path & hyper-parameter configuration for the Titanic task.

All other modules import from here – never hard-code paths elsewhere.
"""
from pathlib import Path

# ── Project root (two levels up from this file: src/titanic/config.py) ──
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

# ── Data directories ─────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# ── Artefact directories ──────────────────────────────────────────────────
MODELS_DIR  = ROOT_DIR / "models"  / "titanic"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "titanic"

# ── File paths ────────────────────────────────────────────────────────────
RAW_DATA_PATH         = RAW_DIR         / "titanic.csv"
ENCODED_DATA_PATH     = PROCESSED_DIR   / "titanic_encoded.csv"
PREDICTIONS_PATH      = PREDICTIONS_DIR / "titanic_predictions.csv"
MODEL_PATH            = MODELS_DIR      / "svc_model.pkl"
EDA_PLOT_PATH         = OUTPUTS_DIR     / "eda.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR     / "confusion_matrix.png"

# ── Feature configuration ─────────────────────────────────────────────────
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET   = "survived"

# ── Categorical columns that need label-encoding ──────────────────────────
CATEGORICAL_FEATURES = ["sex", "embarked"]

# ── Model hyper-parameters ────────────────────────────────────────────────
SVC_PARAMS = {
    "kernel":        "rbf",
    "random_state":  42,
}

TRAIN_TEST_SPLIT_PARAMS = {
    "test_size":    0.2,
    "random_state": 42,
}
