"""
Data loading and pre-processing for the Titanic task.

Responsibilities
----------------
- Load the raw Titanic dataset (seaborn built-in, cached locally).
- Clean and label-encode categorical features.
- Persist the processed dataset to disk.
- Return train/test splits ready for model training.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.titanic import config

log = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Return the raw Titanic DataFrame (seaborn built-in dataset)."""
    log.info("Loading raw Titanic dataset via seaborn …")
    df = sns.load_dataset("titanic")
    log.info("Raw dataset loaded — shape: %s", df.shape)
    return df


def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Clean, encode and split the raw DataFrame.

    Parameters
    ----------
    df : raw Titanic DataFrame from :func:`load_raw`.

    Returns
    -------
    X_train, X_test, y_train, y_test : train/test splits.
    encoders : dict mapping column name → fitted LabelEncoder (for inference).
    """
    log.info("Pre-processing: selecting features and dropping nulls …")
    df_model = df[config.FEATURES + [config.TARGET]].copy()
    before = len(df_model)
    df_model.dropna(inplace=True)
    log.info("Dropped %d rows with nulls; remaining: %d", before - len(df_model), len(df_model))

    encoders: dict[str, LabelEncoder] = {}
    for col in config.CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le
        log.debug("Encoded column '%s' → classes: %s", col, list(le.classes_))

    # Persist encoded dataset for inspection / reproducibility
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(config.ENCODED_DATA_PATH, index=False)
    log.info("Encoded dataset saved → %s", config.ENCODED_DATA_PATH)

    X = df_model[config.FEATURES]
    y = df_model[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, **config.TRAIN_TEST_SPLIT_PARAMS)
    log.info("Train/test split → train: %d | test: %d", len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test, encoders


def save_predictions(df_raw: pd.DataFrame, predictions: list, encoders: dict) -> None:
    """Append predictions to a copy of the raw sample and persist to disk."""
    out = df_raw.copy()
    out["Predicted_Survival"] = ["Survived" if p == 1 else "Did Not Survive" for p in predictions]
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PREDICTIONS_PATH, index=False)
    log.info("Predictions saved → %s", config.PREDICTIONS_PATH)
