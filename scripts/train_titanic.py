#!/usr/bin/env python
"""
scripts/train_titanic.py
========================
CLI entry-point that runs the full Titanic ML pipeline end-to-end.

Usage
-----
    python scripts/train_titanic.py

Steps executed
--------------
1. Load raw data
2. EDA visualisation
3. Pre-process & encode
4. Train SVC
5. Evaluate
6. Plot confusion matrix
7. Run sample predictions
8. Persist model
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Make the project root importable when run as a script ────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.titanic import config, data, model, plots

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_titanic")


def main() -> None:
    print("=" * 60)
    print("  TASK 1 – TITANIC PASSENGER SURVIVAL PREDICTION")
    print("=" * 60)

    # 1 ── Load ────────────────────────────────────────────────────
    log.info("[1/7] Data collection")
    df_raw = data.load_raw()
    log.info("Sample:\n%s", df_raw.head())

    # 2 ── EDA ─────────────────────────────────────────────────────
    log.info("[2/7] Exploratory Data Analysis")
    log.info("Shape: %s | Nulls:\n%s", df_raw.shape, df_raw.isnull().sum().to_string())
    log.info("Survival counts:\n%s", df_raw["survived"].value_counts().to_string())
    plots.plot_eda(df_raw)

    # 3 + 4 ── Pre-process & train ─────────────────────────────────
    log.info("[3/7] Pre-processing & label encoding")
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
        "pclass":   [1,        3,      2],
        "sex":      ["female", "male", "female"],
        "age":      [29,       22,     35],
        "sibsp":    [0,        1,      1],
        "parch":    [0,        0,      0],
        "fare":     [211,      7.25,   26],
        "embarked": ["S",      "S",    "C"],
    })
    sample_enc = sample_raw.copy()
    for col, le in encoders.items():
        sample_enc[col] = le.transform(sample_enc[col])
    preds = clf.predict(sample_enc)
    data.save_predictions(sample_raw, preds, encoders)
    print("\n  Sample predictions:\n", sample_raw.assign(prediction=preds).to_string(index=False))

    # 8 ── Persist model ───────────────────────────────────────────
    model.save(clf)

    print("\n" + "=" * 60)
    print("  TASK 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
