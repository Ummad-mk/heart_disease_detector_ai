"""
====================================================================
Task 1 – Titanic Passenger Survival Prediction System
====================================================================
Deployment link (Streamlit Cloud):
  https://ml-teaching-notebooks-jrrdwercxskw5kwc6rufvp.streamlit.app/

This script reproduces every step of the ML lifecycle:
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
print("TASK 1 – TITANIC PASSENGER SURVIVAL PREDICTION")
print("=" * 60)
print("\n[1] DATA COLLECTION")
print("-" * 40)

# Load Titanic dataset from seaborn (built-in dataset)
df = sns.load_dataset("titanic")
print(f"Dataset loaded. Shape: {df.shape}")
print(df.head())

# ── 2. Data Understanding ─────────────────────────────────────────
print("\n[2] DATA UNDERSTANDING (EDA)")
print("-" * 40)
print("\nColumn Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nNull Values per Column:")
print(df.isnull().sum())
print("\nSurvival value counts:")
print(df["survived"].value_counts())

# EDA Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Titanic – Exploratory Data Analysis", fontsize=14)

# 1) Survival count
sns.countplot(data=df, x="survived", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Survival Count (0=No, 1=Yes)")

# 2) Survival by sex
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0, 1], palette="Set1")
axes[0, 1].set_title("Survival by Sex")

# 3) Survival by class
sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1, 0], palette="Set3")
axes[1, 0].set_title("Survival by Passenger Class")

# 4) Age distribution
df["age"].dropna().hist(bins=30, ax=axes[1, 1], color="steelblue", edgecolor="black")
axes[1, 1].set_title("Age Distribution")

plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=100)
plt.close()
print("EDA plots saved to titanic_eda.png")

# ── 3. Data Pre-processing & Label Encoding ───────────────────────
print("\n[3] DATA PRE-PROCESSING & LABEL ENCODING")
print("-" * 40)

# Select relevant features (same as the repo notebook)
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target   = "survived"

df_model = df[features + [target]].copy()

# Drop rows with missing values
df_model.dropna(inplace=True)
print(f"Shape after dropping nulls: {df_model.shape}")

# Label Encoding for categorical columns
le_sex      = LabelEncoder()
le_embarked = LabelEncoder()

df_model["sex"]      = le_sex.fit_transform(df_model["sex"])
df_model["embarked"] = le_embarked.fit_transform(df_model["embarked"])

print("\nEncoded sample:")
print(df_model.head())

# Save encoded data for reference
df_model.to_csv("titanic_encoded.csv", index=False)
print("Encoded dataset saved to titanic_encoded.csv")

# ── 4. Train / Test Split & Model Training ────────────────────────
print("\n[4] MODEL TRAINING – Support Vector Classifier (SVC)")
print("-" * 40)

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

svc = SVC(kernel="rbf", random_state=42)
svc.fit(X_train, y_train)
print("Model trained successfully.")

# ── 5. Model Evaluation ───────────────────────────────────────────
print("\n[5] MODEL EVALUATION")
print("-" * 40)

y_pred = svc.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"Accuracy : {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Titanic – Confusion Matrix")
plt.tight_layout()
plt.savefig("titanic_confusion_matrix.png", dpi=100)
plt.close()
print("Confusion matrix saved to titanic_confusion_matrix.png")

# ── 6. Making Predictions on Sample Data ─────────────────────────
print("\n[6] MAKING PREDICTIONS ON SAMPLE DATA")
print("-" * 40)

# Sample passengers
sample_raw = pd.DataFrame({
    "pclass":   [1,    3,    2],
    "sex":      ["female", "male", "female"],
    "age":      [29,   22,   35],
    "sibsp":    [0,    1,    1],
    "parch":    [0,    0,    0],
    "fare":     [211,  7.25, 26],
    "embarked": ["S",  "S",  "C"],
})
sample = sample_raw.copy()
sample["sex"]      = le_sex.transform(sample["sex"])
sample["embarked"] = le_embarked.transform(sample["embarked"])

predictions = svc.predict(sample)
sample_raw["Predicted_Survival"] = ["Survived" if p == 1 else "Did Not Survive"
                                    for p in predictions]
print(sample_raw.to_string(index=False))

# Save predictions
sample_raw.to_csv("titanic_predictions.csv", index=False)
print("\nPredictions saved to titanic_predictions.csv")

# ── 7. Save Trained Model ─────────────────────────────────────────
print("\n[7] SAVING TRAINED MODEL")
print("-" * 40)

with open("titanic_svc_model.pkl", "wb") as f:
    pickle.dump(svc, f)
print("Model saved to titanic_svc_model.pkl")

print("\n" + "=" * 60)
print("TASK 1 COMPLETE")
print("=" * 60)
