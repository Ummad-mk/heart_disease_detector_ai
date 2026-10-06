"""
Modal Deployment – Heart Disease Prediction App
================================================
Deploys the Heart Disease Streamlit app as a persistent
web endpoint on Modal's serverless infrastructure.

Usage:
  modal deploy modal_heart_disease.py        # deploy permanently
  modal serve  modal_heart_disease.py        # ephemeral (dev mode)

After deploying, Modal gives you a public HTTPS URL like:
  https://ummad-mk--heart-disease-predictor-run.modal.run
"""

import modal
from pathlib import Path

# ── App definition ────────────────────────────────────────────────
app = modal.App("heart-disease-predictor")

# ── Container image with all dependencies ─────────────────────────
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

# ── Mount local project files into the container ──────────────────
LOCAL_DIR = Path(__file__).parent
project_mount = modal.Mount.from_local_dir(
    LOCAL_DIR,
    remote_path="/app",
    # Only copy what the app actually needs
    condition=lambda p: any(
        p.endswith(ext)
        for ext in [
            "heart_disease_app.py",
            "heart_disease_svc_model.pkl",
            "heart_disease_le_cp.pkl",
            "heart.csv",
        ]
    ),
)

# ── Web endpoint ──────────────────────────────────────────────────
@app.function(
    image=image,
    mounts=[project_mount],
    allow_concurrent_inputs=10,
    # Keep one container warm to avoid cold starts
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
            "/app/heart_disease_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ]
    )
