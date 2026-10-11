"""
Visualisation helpers for the Heart Disease task.

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

from src.heart_disease import config

log = logging.getLogger(__name__)


def plot_eda(df: pd.DataFrame) -> None:
    """
    Save a 2×2 EDA figure to :data:`config.EDA_PLOT_PATH`.

    Panels
    ------
    1. Target distribution
    2. Age distribution by target
    3. Max heart rate (thalach) by target
    4. Chest pain type by target
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Heart Disease – Exploratory Data Analysis", fontsize=14, fontweight="bold")

    sns.countplot(data=df, x="target", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Heart Disease Count (0=No, 1=Yes)")

    df[df["target"] == 0]["age"].hist(bins=20, ax=axes[0, 1], alpha=0.7, color="green", label="No Disease")
    df[df["target"] == 1]["age"].hist(bins=20, ax=axes[0, 1], alpha=0.7, color="red",   label="Disease")
    axes[0, 1].set_title("Age Distribution by Target")
    axes[0, 1].legend()

    sns.boxplot(data=df, x="target", y="thalach", ax=axes[1, 0], palette="Set1")
    axes[1, 0].set_title("Max Heart Rate by Target")
    axes[1, 0].set_xlabel("Target (0=No, 1=Yes)")

    sns.countplot(data=df, x="cp", hue="target", ax=axes[1, 1], palette="Set3")
    axes[1, 1].set_title("Chest Pain Type by Target")

    plt.tight_layout()
    plt.savefig(config.EDA_PLOT_PATH, dpi=100)
    plt.close(fig)
    log.info("EDA plot saved → %s", config.EDA_PLOT_PATH)


def plot_confusion_matrix(cm: np.ndarray) -> None:
    """Save a labelled confusion-matrix heatmap to :data:`config.CONFUSION_MATRIX_PATH`."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Heart Disease – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=100)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", config.CONFUSION_MATRIX_PATH)


def plot_correlation(df: pd.DataFrame) -> None:
    """Save a full feature-correlation heatmap to :data:`config.CORRELATION_PLOT_PATH`."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heart Disease – Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(config.CORRELATION_PLOT_PATH, dpi=100)
    plt.close(fig)
    log.info("Correlation heatmap saved → %s", config.CORRELATION_PLOT_PATH)
