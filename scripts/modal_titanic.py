"""
Modal Deployment – Titanic Survival Prediction App
===================================================
Deploy: modal deploy scripts/modal_titanic.py
Dev:    modal serve  scripts/modal_titanic.py
"""

import modal
from pathlib import Path

app = modal.App("titanic-predictor")

# Project root is one level above scripts/
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Build image: install deps AND bake project files into container ─
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit==1.45.1",
        "scikit-learn==1.9.1",
        "pandas==2.2.3",
        "numpy==2.2.4",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "matplotlib==3.9.2",
        "joblib==1.4.2",
        "threadpoolctl==3.5.0",
        "narwhals>=2.0.1",
    )
    .add_local_dir(str(ROOT_DIR / "src"), remote_path="/app/src", copy=True)
    .add_local_dir(str(ROOT_DIR / "apps"), remote_path="/app/apps", copy=True)
    .add_local_dir(str(ROOT_DIR / "models"), remote_path="/app/models", copy=True)
    .add_local_dir(str(ROOT_DIR / "data"), remote_path="/app/data", copy=True)
)

# ── Web endpoint ──────────────────────────────────────────────────
@app.function(
    image=image,
    timeout=300,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8501, startup_timeout=60)
def run():
    import os, subprocess, sys
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "/app/apps/titanic_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ], cwd="/app", env=env)
