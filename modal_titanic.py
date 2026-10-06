"""
Modal Deployment – Titanic Survival Prediction App
===================================================
Deploys the Titanic Streamlit app as a persistent
web endpoint on Modal's serverless infrastructure.

Usage:
  modal deploy modal_titanic.py        # deploy permanently
  modal serve  modal_titanic.py        # ephemeral (dev mode)

After deploying, Modal gives you a public HTTPS URL like:
  https://ummad-mk--titanic-predictor-run.modal.run
"""

import modal
from pathlib import Path

# ── App definition ────────────────────────────────────────────────
app = modal.App("titanic-predictor")

# ── Container image ───────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit==1.45.1",
        "scikit-learn==1.9.1",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "seaborn==0.13.2",
        "matplotlib==3.9.2",
        "joblib==1.4.2",
        "threadpoolctl==3.5.0",
        "narwhals>=2.0.1",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# ── Mount only what the app needs ─────────────────────────────────
LOCAL_DIR = Path(__file__).parent
project_mount = modal.Mount.from_local_dir(
    LOCAL_DIR,
    remote_path="/app",
    condition=lambda p: any(
        p.endswith(ext)
        for ext in [
            "titanic_app.py",
            "titanic_svc_model.pkl",
            "titanic.csv",
        ]
    ),
)

# ── Web endpoint ──────────────────────────────────────────────────
@app.function(
    image=image,
    mounts=[project_mount],
    allow_concurrent_inputs=10,
    min_containers=1,
    timeout=300,
)
@modal.web_server(port=8501, startup_timeout=60)
def run():
    import subprocess
    import sys

    subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "/app/titanic_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ]
    )
