"""
Script to generate both Jupyter notebooks for the assignment.
Run this script to create:
  - Task1_Titanic_Survival_Prediction.ipynb
  - Task2_Heart_Disease_Prediction.ipynb
"""

import json

# ────────────────────────────────────────────────────────────────────
# Helper to build a notebook cell
# ────────────────────────────────────────────────────────────────────
def code_cell(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source if isinstance(source, list) else [source]
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def notebook(cells, kernel_name="python3"):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernel_name
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

# ════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 – TITANIC PASSENGER SURVIVAL PREDICTION
# ════════════════════════════════════════════════════════════════════
titanic_cells = [

    md_cell("""# Assignment 2 – Task 1: Titanic Passenger Survival Prediction System
---
**Deployment Link:** [https://ml-teaching-notebooks-jrrdwercxskw5kwc6rufvp.streamlit.app/](https://ml-teaching-notebooks-jrrdwercxskw5kwc6rufvp.streamlit.app/)

This notebook follows the complete Machine Learning lifecycle:
1. Data Collection
2. Data Understanding (EDA)
3. Data Pre-processing & Label Encoding
4. Model Training (SVC)
5. Model Evaluation
6. Making Predictions
7. Saving the Trained Model

**Dataset:** Titanic (built-in seaborn dataset)  
**Algorithm:** Support Vector Classifier (SVC)
"""),

    md_cell("## Step 0 – Import Libraries"),
    code_cell("""\
import os, pickle, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")
print("Libraries imported successfully OK")"""),

    md_cell("## Step 1 – Data Collection\nLoad the Titanic dataset (seaborn built-in)."),
    code_cell("""\
df = sns.load_dataset("titanic")
print(f"Dataset Shape: {df.shape}")
df.head(10)"""),

    md_cell("## Step 2 – Data Understanding (EDA)"),
    code_cell("""\
print("=== Dataset Info ===")
df.info()"""),

    code_cell("""\
print("=== Descriptive Statistics ===")
df.describe()"""),

    code_cell("""\
print("=== Missing Values ===")
df.isnull().sum()"""),

    code_cell("""\
print("=== Survival Distribution ===")
print(df["survived"].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Titanic – Exploratory Data Analysis", fontsize=14, fontweight="bold")

sns.countplot(data=df, x="survived", ax=axes[0,0], palette="Set2")
axes[0,0].set_title("Survival Count (0=No, 1=Yes)")

sns.countplot(data=df, x="sex", hue="survived", ax=axes[0,1], palette="Set1")
axes[0,1].set_title("Survival by Sex")

sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1,0], palette="Set3")
axes[1,0].set_title("Survival by Passenger Class")

df["age"].dropna().hist(bins=30, ax=axes[1,1], color="steelblue", edgecolor="black")
axes[1,1].set_title("Age Distribution")

plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=100)
plt.show()
print("EDA plots saved OK")"""),

    md_cell("## Step 3 – Data Pre-processing & Label Encoding"),
    code_cell("""\
# Select relevant features
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target   = "survived"

df_model = df[features + [target]].copy()
df_model.dropna(inplace=True)
print(f"Shape after dropping nulls: {df_model.shape}")

# Label Encoding
le_sex      = LabelEncoder()
le_embarked = LabelEncoder()

df_model["sex"]      = le_sex.fit_transform(df_model["sex"])
df_model["embarked"] = le_embarked.fit_transform(df_model["embarked"])

# Save encoded dataset
df_model.to_csv("titanic_encoded.csv", index=False)
print("Encoded dataset saved to titanic_encoded.csv OK")
df_model.head()"""),

    md_cell("## Step 4 – Model Training (Support Vector Classifier)"),
    code_cell("""\
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

svc = SVC(kernel="rbf", random_state=42)
svc.fit(X_train, y_train)
print("\\nSVC model trained successfully OK")"""),

    md_cell("## Step 5 – Model Evaluation"),
    code_cell("""\
y_pred = svc.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print("\\nConfusion Matrix:")
print(cm)
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive","Survived"],
            yticklabels=["Did Not Survive","Survived"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Titanic – Confusion Matrix")
plt.tight_layout()
plt.savefig("titanic_confusion_matrix.png", dpi=100)
plt.show()"""),

    md_cell("## Step 6 – Making Predictions on Sample Data"),
    code_cell("""\
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
sample_raw["Predicted_Survival"] = ["Survived ✓" if p==1 else "Did Not Survive ✗"
                                     for p in predictions]

# Save predictions
sample_raw.to_csv("titanic_predictions.csv", index=False)
print("Predictions saved to titanic_predictions.csv OK")
sample_raw"""),

    md_cell("## Step 7 – Save Trained Model"),
    code_cell("""\
with open("titanic_svc_model.pkl", "wb") as f:
    pickle.dump(svc, f)

print("Model saved to titanic_svc_model.pkl OK")
print("\\n" + "="*50)
print("TASK 1 COMPLETE")
print("="*50)"""),
]

# ════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 – HEART DISEASE PREDICTION
# ════════════════════════════════════════════════════════════════════
heart_cells = [

    md_cell("""# Assignment 2 – Task 2: Heart Disease Prediction System
---
**Deployment Link:** [https://heart-disease-predictor-ml.streamlit.app/](https://heart-disease-predictor-ml.streamlit.app/)

This notebook follows the **exact same steps** as Task 1 (Titanic Prediction):
1. Data Collection
2. Data Understanding (EDA)
3. Data Pre-processing & Label Encoding
4. Model Training (SVC)
5. Model Evaluation
6. Making Predictions
7. Saving the Trained Model

**Dataset:** UCI Heart Disease (Cleveland) – `heart.csv`  
**Algorithm:** Support Vector Classifier (SVC)  
**Constraint:** Only 4 input attributes selected from the dataset

### Selected 4 Input Attributes:
| Feature | Description |
|---------|-------------|
| `age` | Age of the patient (years) |
| `thalach` | Maximum heart rate achieved |
| `chol` | Serum cholesterol (mg/dl) |
| `cp` | Chest pain type (0-3) |
"""),

    md_cell("## Step 0 – Import Libraries"),
    code_cell("""\
import os, pickle, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")
print("Libraries imported successfully OK")"""),

    md_cell("## Step 1 – Data Collection\nLoad the Heart Disease dataset from the local CSV file."),
    code_cell("""\
df = pd.read_csv("heart.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head(10)"""),

    md_cell("## Step 2 – Data Understanding (EDA)"),
    code_cell("""\
print("=== Dataset Info ===")
df.info()"""),

    code_cell("""\
print("=== Descriptive Statistics ===")
df.describe()"""),

    code_cell("""\
print("=== Missing Values ===")
df.isnull().sum()"""),

    code_cell("""\
print("=== Target Distribution (0=No Disease, 1=Disease) ===")
print(df["target"].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Heart Disease Dataset – Exploratory Data Analysis", fontsize=14, fontweight="bold")

sns.countplot(data=df, x="target", ax=axes[0,0], palette="Set2")
axes[0,0].set_title("Heart Disease Count (0=No, 1=Yes)")

df[df["target"]==0]["age"].hist(bins=20, ax=axes[0,1], alpha=0.7, color="green", label="No Disease")
df[df["target"]==1]["age"].hist(bins=20, ax=axes[0,1], alpha=0.7, color="red", label="Disease")
axes[0,1].set_title("Age Distribution by Target")
axes[0,1].legend()

sns.boxplot(data=df, x="target", y="thalach", ax=axes[1,0], palette="Set1")
axes[1,0].set_title("Max Heart Rate by Target")
axes[1,0].set_xlabel("Target (0=No, 1=Yes)")

sns.countplot(data=df, x="cp", hue="target", ax=axes[1,1], palette="Set3")
axes[1,1].set_title("Chest Pain Type by Target")

plt.tight_layout()
plt.savefig("heart_disease_eda.png", dpi=100)
plt.show()
print("EDA plots saved OK")"""),

    code_cell("""\
# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heart Disease – Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heart_disease_correlation.png", dpi=100)
plt.show()
print("Correlation heatmap saved OK")"""),

    md_cell("""## Step 3 – Data Pre-processing & Label Encoding

### ⚠️ Constraint: Select only 4 input attributes
Selected features: **age**, **thalach**, **chol**, **cp**"""),
    code_cell("""\
# === SELECTED 4 FEATURES (Assignment Constraint) ===
SELECTED_FEATURES = ["age", "thalach", "chol", "cp"]
TARGET = "target"

print(f"Selected features: {SELECTED_FEATURES}")
print(f"Target: {TARGET}\\n")

df_model = df[SELECTED_FEATURES + [TARGET]].copy()

# Handle missing values
print(f"Missing values:\\n{df_model.isnull().sum()}")
df_model.dropna(inplace=True)
print(f"\\nShape after handling nulls: {df_model.shape}")

# Label Encoding for chest pain type (cp)
le_cp = LabelEncoder()
df_model["cp"] = le_cp.fit_transform(df_model["cp"])
print(f"\\nLabel Encoder classes for cp: {le_cp.classes_}")

# Save encoded dataset
df_model.to_csv("heart_disease_encoded.csv", index=False)
print("Encoded dataset saved to heart_disease_encoded.csv OK")
df_model.head()"""),

    md_cell("## Step 4 – Model Training (Support Vector Classifier)"),
    code_cell("""\
X = df_model[SELECTED_FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

svc_heart = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svc_heart.fit(X_train, y_train)
print("\\nSVC model trained successfully OK")"""),

    md_cell("## Step 5 – Model Evaluation"),
    code_cell("""\
y_pred = svc_heart.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print("\\nConfusion Matrix:")
print(cm)
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease","Disease"]))

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Disease","Disease"],
            yticklabels=["No Disease","Disease"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Heart Disease – Confusion Matrix")
plt.tight_layout()
plt.savefig("heart_disease_confusion_matrix.png", dpi=100)
plt.show()"""),

    md_cell("## Step 6 – Making Predictions on Sample Data"),
    code_cell("""\
sample_data = pd.DataFrame({
    "age":     [52,   67,   45,   38,   60],
    "thalach": [168,  108,  150,  170,  120],
    "chol":    [212,  300,  225,  174,  280],
    "cp":      [0,    1,    2,    3,    0],
})
sample_encoded = sample_data.copy()
sample_encoded["cp"] = le_cp.transform(sample_encoded["cp"])

predictions = svc_heart.predict(sample_encoded)
sample_data["Predicted"] = ["❤️ Heart Disease" if p==1 else "✅ No Heart Disease"
                              for p in predictions]

# Save predictions
sample_data.to_csv("heart_disease_predictions.csv", index=False)
print("Predictions saved to heart_disease_predictions.csv OK")
sample_data"""),

    md_cell("## Step 7 – Save Trained Model"),
    code_cell("""\
# Save the SVC model
with open("heart_disease_svc_model.pkl", "wb") as f:
    pickle.dump(svc_heart, f)

# Save the LabelEncoder for 'cp'
with open("heart_disease_le_cp.pkl", "wb") as f:
    pickle.dump(le_cp, f)

print("Model saved to heart_disease_svc_model.pkl OK")
print("Label encoder saved to heart_disease_le_cp.pkl OK")
print("\\n" + "="*50)
print("TASK 2 COMPLETE")
print("="*50)"""),
]

# ── Write notebooks ──────────────────────────────────────────────
with open("Task1_Titanic_Survival_Prediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook(titanic_cells), f, indent=1)
print("Created: Task1_Titanic_Survival_Prediction.ipynb OK")

with open("Task2_Heart_Disease_Prediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook(heart_cells), f, indent=1)
print("Created: Task2_Heart_Disease_Prediction.ipynb OK")
