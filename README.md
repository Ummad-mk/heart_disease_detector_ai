# ML Assignment 2 – Titanic & Heart Disease Prediction

> Two end-to-end supervised ML pipelines using Support Vector Classifiers (SVC).

---

## Project Structure

```
ml-assignment/
│
├── src/                        # Importable Python packages
│   ├── titanic/
│   │   ├── config.py           # All paths & hyper-params (single source of truth)
│   │   ├── data.py             # Load, clean, encode, split
│   │   ├── model.py            # Train, evaluate, save, load
│   │   └── plots.py            # EDA & confusion matrix figures
│   └── heart_disease/
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       └── plots.py            # + correlation heatmap
│
├── scripts/                    # CLI entry-points (thin wrappers over src/)
│   ├── train_titanic.py
│   ├── train_heart_disease.py
│   ├── modal_titanic.py        # Modal.com serverless deploy
│   └── modal_heart_disease.py
│
├── apps/                       # Streamlit web applications
│   ├── titanic_app.py
│   └── heart_disease_app.py
│
├── data/
│   ├── raw/                    # Original, immutable source data
│   │   ├── titanic.csv
│   │   └── heart.csv
│   ├── processed/              # Label-encoded datasets (generated)
│   └── predictions/            # Sample prediction CSVs (generated)
│
├── models/
│   ├── titanic/                # titanic/svc_model.pkl
│   └── heart_disease/          # heart_disease/svc_model.pkl + le_cp.pkl
│
├── outputs/
│   ├── titanic/                # eda.png, confusion_matrix.png
│   └── heart_disease/          # eda.png, confusion_matrix.png, correlation.png
│
├── notebooks/
│   ├── 01_titanic_survival_prediction.ipynb
│   └── 02_heart_disease_prediction.ipynb
│
├── tests/                      # pytest test suite
│   ├── titanic/
│   └── heart_disease/
│
├── .streamlit/config.toml      # Streamlit theme
├── .env.example                # Environment variable template
├── .gitignore
├── Makefile                    # Unix/WSL commands
├── run_assignment.bat          # Windows commands
├── pyproject.toml              # Project metadata & packaging
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train models

```bash
# Task 1: Titanic
python scripts/train_titanic.py

# Task 2: Heart Disease
python scripts/train_heart_disease.py
```

Or with Make:
```bash
make train-all
```

### 3 — Launch Streamlit apps

```bash
# Task 1
streamlit run apps/titanic_app.py

# Task 2
streamlit run apps/heart_disease_app.py
```

Or with Make:
```bash
make app-titanic
make app-heart-disease
```

### 4 — Windows (no WSL)

Double-click **`run_assignment.bat`** and choose from the menu.

---

## Architecture

Each task follows an identical layered architecture:

```
scripts/train_*.py          ← CLI orchestrator (no ML logic)
    │
    ├── src/*/config.py     ← Paths, feature lists, hyper-params
    ├── src/*/data.py       ← I/O, cleaning, encoding, splitting
    ├── src/*/model.py      ← Training, evaluation, persistence
    └── src/*/plots.py      ← Visualisation (pure side-effects)
```

---

## ML Pipeline Steps

| # | Step | Module |
|---|------|--------|
| 1 | Data Collection | `data.load_raw()` |
| 2 | Exploratory Data Analysis | `plots.plot_eda()` |
| 3 | Pre-processing & Label Encoding | `data.preprocess()` |
| 4 | Model Training (SVC RBF) | `model.train()` |
| 5 | Evaluation | `model.evaluate()` |
| 6 | Confusion Matrix | `plots.plot_confusion_matrix()` |
| 7 | Sample Predictions | `data.save_predictions()` |
| 8 | Persist Model | `model.save()` |

---

## Datasets

| Task | Dataset | Source | Size |
|------|---------|--------|------|
| 1 – Titanic | Titanic Passengers | Seaborn built-in | 891 rows |
| 2 – Heart Disease | UCI Heart Disease (Cleveland) | `data/raw/heart.csv` | 303 rows |

---

## Running Tests

```bash
pytest tests/ -v
# or
make test
```
