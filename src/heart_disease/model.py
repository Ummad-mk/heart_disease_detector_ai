"""
Model training, evaluation and persistence for the Heart Disease task.

Responsibilities
----------------
- Train an SVC classifier on the prepared data.
- Evaluate with accuracy, confusion matrix and classification report.
- Save / load the trained model and label encoder to / from disk (pickle).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.heart_disease import config

log = logging.getLogger(__name__)


# ── Training ──────────────────────────────────────────────────────────────

def train(X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    """Fit an SVC on the training data and return the fitted estimator."""
    log.info("Training SVC — params: %s", config.SVC_PARAMS)
    model = SVC(**config.SVC_PARAMS)
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(
    model: SVC,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, np.ndarray, str]:
    """
    Evaluate a fitted model on the held-out test set.

    Returns
    -------
    accuracy : float
    cm       : confusion matrix ndarray
    report   : full classification report string
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])

    log.info("Accuracy: %.2f%%", acc * 100)
    log.info("Confusion Matrix:\n%s", cm)
    log.info("Classification Report:\n%s", report)
    return acc, cm, report


# ── Persistence ───────────────────────────────────────────────────────────

def save(model: SVC, encoders: dict) -> Path:
    """Pickle the model and label encoders to their configured paths."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    log.info("Model saved → %s", config.MODEL_PATH)

    with open(config.ENCODER_PATH, "wb") as fh:
        pickle.dump(encoders, fh)
    log.info("Encoders saved → %s", config.ENCODER_PATH)

    return config.MODEL_PATH


def load() -> Tuple[SVC, dict]:
    """Load and return the pickled SVC and encoders from their configured paths."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. "
            "Run `scripts/train_heart_disease.py` first."
        )
    with open(config.MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    with open(config.ENCODER_PATH, "rb") as fh:
        encoders = pickle.load(fh)
    log.info("Model and encoders loaded ← %s", config.MODELS_DIR)
    return model, encoders
