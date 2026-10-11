#!/usr/bin/env python
"""
scripts/train_heart_disease.py
==============================
CLI entry-point that runs the full Heart Disease ML pipeline end-to-end.

Usage
-----
    python scripts/train_heart_disease.py

Steps executed
--------------
1. Load raw data
2. EDA visualisation (4-panel + correlation heatmap)
3. Pre-process & encode (4 selected features)
4. Train SVC
5. Evaluate
6. Plot confusion matrix
7. Run sample predictions
8. Persist model & encoders
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Make the project root importable when run as a script ────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heart_disease import config, data, model, plots

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_heart_disease")


def main() -> None:
    print("=" * 60)
    print("  TASK 2 – HEART DISEASE PREDICTION SYSTEM")
    print("=" * 60)

    # 1 ── Load ────────────────────────────────────────────────────
    log.info("[1/7] Data collection")
    df_raw = data.load_raw()
    log.info("Sample:\n%s", df_raw.head())

    # 2 ── EDA ─────────────────────────────────────────────────────
    log.info("[2/7] Exploratory Data Analysis")
    log.info("Shape: %s | Nulls:\n%s", df_raw.shape, df_raw.isnull().sum().to_string())
    log.info("Target distribution:\n%s", df_raw["target"].value_counts().to_string())
    plots.plot_eda(df_raw)
    plots.plot_correlation(df_raw)

    # 3 + 4 ── Pre-process & train ─────────────────────────────────
    log.info("[3/7] Pre-processing & label encoding")
    log.info("Selected features (assignment constraint – 4 attributes): %s", config.FEATURES)
    X_train, X_test, y_train, y_test, encoders = data.preprocess(df_raw)

    log.info("[4/7] Model training")
    clf = model.train(X_train, y_train)

    # 5 ── Evaluate ────────────────────────────────────────────────
    log.info("[5/7] Evaluation")
    acc, cm, report = model.evaluate(clf, X_test, y_test)
    print(f"\n  Accuracy : {acc * 100:.2f}%")
    print(f"\n  Classification Report:\n{report}")

    # 6 ── Confusion matrix ────────────────────────────────────────
    log.info("[6/7] Confusion matrix plot")
    plots.plot_confusion_matrix(cm)

    # 7 ── Sample predictions ──────────────────────────────────────
    log.info("[7/7] Sample predictions")
    sample_raw = pd.DataFrame({
        "age":     [52,  67,  45,  38,  60],
        "thalach": [168, 108, 150, 170, 120],
        "chol":    [212, 300, 225, 174, 280],
        "cp":      [0,   1,   2,   3,   0],
    })
    sample_enc = sample_raw.copy()
    for col, le in encoders.items():
        sample_enc[col] = le.transform(sample_enc[col])
    preds = clf.predict(sample_enc)
    data.save_predictions(sample_raw, preds)
    print("\n  Sample predictions:\n", sample_raw.assign(prediction=preds).to_string(index=False))

    # 8 ── Persist model & encoders ────────────────────────────────
    model.save(clf, encoders)

    print("\n" + "=" * 60)
    print("  TASK 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
