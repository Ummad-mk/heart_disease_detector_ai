"""
Visualisation helpers for the Titanic task.

Each function is a pure side-effect: it creates a plot, saves it to the
configured output path, and closes the figure. No return value.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.titanic import config

log = logging.getLogger(__name__)


def plot_eda(df: pd.DataFrame) -> None:
    """
    Save a 2×2 exploratory-data-analysis figure to :data:`config.EDA_PLOT_PATH`.

    Panels
    ------
    1. Survival count
    2. Survival by sex
    3. Survival by passenger class
    4. Age distribution
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Titanic – Exploratory Data Analysis", fontsize=14, fontweight="bold")

    sns.countplot(data=df, x="survived", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Survival Count (0=No, 1=Yes)")

    sns.countplot(data=df, x="sex", hue="survived", ax=axes[0, 1], palette="Set1")
    axes[0, 1].set_title("Survival by Sex")

    sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1, 0], palette="Set3")
    axes[1, 0].set_title("Survival by Passenger Class")

    df["age"].dropna().hist(bins=30, ax=axes[1, 1], color="steelblue", edgecolor="black")
    axes[1, 1].set_title("Age Distribution")

    plt.tight_layout()
    plt.savefig(config.EDA_PLOT_PATH, dpi=100)
    plt.close(fig)
    log.info("EDA plot saved → %s", config.EDA_PLOT_PATH)


def plot_confusion_matrix(cm: np.ndarray) -> None:
    """Save a labelled confusion-matrix heatmap to :data:`config.CONFUSION_MATRIX_PATH`."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Titanic – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=100)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", config.CONFUSION_MATRIX_PATH)
