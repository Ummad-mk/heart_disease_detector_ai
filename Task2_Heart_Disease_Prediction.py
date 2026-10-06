"""
====================================================================
Task 2 – Heart Disease Prediction System
====================================================================
Deployment link (Streamlit Cloud):
  https://heart-disease-predictor-ml.streamlit.app/
  (deploy heart_disease_app.py via Streamlit Community Cloud)

Dataset: UCI Heart Disease (Cleveland)
Source : https://archive.ics.uci.edu/dataset/45/heart+disease
Local  : heart.csv  (already in project folder)

Selected 4 Input Attributes (from 13 available):
  1. age      – Age of the patient (years)
  2. thalach  – Maximum heart rate achieved
  3. chol     – Serum cholesterol (mg/dl)
  4. cp       – Chest pain type (0-3)

Steps follow the EXACT same structure as Task 1 (Titanic):
  1. Data Collection
  2. Data Understanding (EDA)
  3. Data Pre-processing & Label Encoding
  4. Model Training (SVC)
  5. Model Evaluation
  6. Making Predictions
  7. Saving the trained model
====================================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────
import os
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – no window needed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")

# ── 1. Data Collection ────────────────────────────────────────────
print("=" * 60)
print("TASK 2 – HEART DISEASE PREDICTION SYSTEM")
print("=" * 60)
print("\n[1] DATA COLLECTION")
print("-" * 40)

df = pd.read_csv("heart.csv")
print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ── 2. Data Understanding ─────────────────────────────────────────
print("\n[2] DATA UNDERSTANDING (EDA)")
print("-" * 40)
print("\nColumn Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
print("\nNull Values per Column:")
print(df.isnull().sum())
print("\nTarget distribution (0=No Disease, 1=Disease):")
print(df["target"].value_counts())

# EDA Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Heart Disease Dataset – Exploratory Data Analysis", fontsize=14)

# 1) Target distribution
sns.countplot(data=df, x="target", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Heart Disease Count (0=No, 1=Yes)")

# 2) Age distribution by target
df[df["target"] == 0]["age"].hist(bins=20, ax=axes[0, 1], alpha=0.7,
                                   color="green", label="No Disease")
df[df["target"] == 1]["age"].hist(bins=20, ax=axes[0, 1], alpha=0.7,
                                   color="red", label="Disease")
axes[0, 1].set_title("Age Distribution by Target")
axes[0, 1].legend()

# 3) Max Heart Rate vs Target
sns.boxplot(data=df, x="target", y="thalach", ax=axes[1, 0], palette="Set1")
axes[1, 0].set_title("Max Heart Rate by Target")
axes[1, 0].set_xlabel("Target (0=No, 1=Yes)")

# 4) Chest Pain type by Target
sns.countplot(data=df, x="cp", hue="target", ax=axes[1, 1], palette="Set3")
axes[1, 1].set_title("Chest Pain Type by Target")

plt.tight_layout()
plt.savefig("heart_disease_eda.png", dpi=100)
plt.close()
print("EDA plots saved to heart_disease_eda.png")

# Correlation heatmap for all features
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heart Disease – Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heart_disease_correlation.png", dpi=100)
plt.close()
print("Correlation heatmap saved to heart_disease_correlation.png")

# ── 3. Data Pre-processing & Label Encoding ───────────────────────
print("\n[3] DATA PRE-PROCESSING & LABEL ENCODING")
print("-" * 40)

# ── 4 SELECTED INPUT ATTRIBUTES (constraint from assignment) ──────
SELECTED_FEATURES = ["age", "thalach", "chol", "cp"]
TARGET = "target"

print(f"\nSelected features: {SELECTED_FEATURES}")
print("(Constraint: only 4 input attributes from Heart Disease dataset)\n")

df_model = df[SELECTED_FEATURES + [TARGET]].copy()

# Check missing values
print(f"Missing values:\n{df_model.isnull().sum()}")
df_model.dropna(inplace=True)
print(f"\nShape after handling nulls: {df_model.shape}")

# Label Encoding for categorical feature 'cp' (chest pain type 0-3)
# cp is already numeric, but we apply LabelEncoder to mirror the Titanic approach
le_cp = LabelEncoder()
df_model["cp"] = le_cp.fit_transform(df_model["cp"])

print("\nEncoded sample (first 5 rows):")
print(df_model.head())

# Save encoded dataset
df_model.to_csv("heart_disease_encoded.csv", index=False)
print("\nEncoded dataset saved to heart_disease_encoded.csv")

# ── 4. Train / Test Split & Model Training ────────────────────────
print("\n[4] MODEL TRAINING – Support Vector Classifier (SVC)")
print("-" * 40)

X = df_model[SELECTED_FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

svc_heart = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svc_heart.fit(X_train, y_train)
print("Model trained successfully.")

# ── 5. Model Evaluation ───────────────────────────────────────────
print("\n[5] MODEL EVALUATION")
print("-" * 40)

y_pred = svc_heart.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"Accuracy : {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Heart Disease – Confusion Matrix")
plt.tight_layout()
plt.savefig("heart_disease_confusion_matrix.png", dpi=100)
plt.close()
print("Confusion matrix saved to heart_disease_confusion_matrix.png")

# ── 6. Making Predictions on Sample Data ─────────────────────────
print("\n[6] MAKING PREDICTIONS ON SAMPLE DATA")
print("-" * 40)

sample_data = pd.DataFrame({
    "age":     [52, 67, 45, 38, 60],
    "thalach": [168, 108, 150, 170, 120],
    "chol":    [212, 300, 225, 174, 280],
    "cp":      [0,   1,   2,   3,   0],
})
sample_encoded = sample_data.copy()
sample_encoded["cp"] = le_cp.transform(sample_encoded["cp"])

predictions = svc_heart.predict(sample_encoded)
sample_data["Predicted"] = ["Heart Disease" if p == 1 else "No Heart Disease"
                             for p in predictions]
print(sample_data.to_string(index=False))

# Save predictions
sample_data.to_csv("heart_disease_predictions.csv", index=False)
print("\nPredictions saved to heart_disease_predictions.csv")

# ── 7. Save Trained Model ─────────────────────────────────────────
print("\n[7] SAVING TRAINED MODEL")
print("-" * 40)

MODEL_FILE = "heart_disease_svc_model.pkl"
with open(MODEL_FILE, "wb") as f:
    pickle.dump(svc_heart, f)

# Also save the label encoder
LE_FILE = "heart_disease_le_cp.pkl"
with open(LE_FILE, "wb") as f:
    pickle.dump(le_cp, f)

print(f"Model saved to {MODEL_FILE}")
print(f"Label encoder saved to {LE_FILE}")

print("\n" + "=" * 60)
print("TASK 2 COMPLETE")
print("=" * 60)
