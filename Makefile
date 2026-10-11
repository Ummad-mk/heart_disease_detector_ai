# =============================================================
# Makefile – ML Assignment 2
# =============================================================
# Works on Linux / macOS / WSL.
# On native Windows use the `run_assignment.bat` helper instead.
# =============================================================

.PHONY: install train-titanic train-heart-disease train-all \
        app-titanic app-heart-disease test lint clean

# ── Environment ───────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Training ──────────────────────────────────────────────────
train-titanic:
	python scripts/train_titanic.py

train-heart-disease:
	python scripts/train_heart_disease.py

train-all: train-titanic train-heart-disease

# ── Apps ──────────────────────────────────────────────────────
app-titanic:
	streamlit run apps/titanic_app.py

app-heart-disease:
	streamlit run apps/heart_disease_app.py

# ── Tests ─────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Code quality ──────────────────────────────────────────────
lint:
	python -m py_compile src/titanic/*.py src/heart_disease/*.py scripts/*.py apps/*.py

# ── Notebooks ─────────────────────────────────────────────────
notebooks:
	jupyter nbconvert --to notebook --execute --inplace \
	  notebooks/01_titanic_survival_prediction.ipynb \
	  notebooks/02_heart_disease_prediction.ipynb

# ── Clean generated artefacts ─────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
