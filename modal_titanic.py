"""
Modal Deployment – Titanic Survival Prediction App
===================================================
modal deploy modal_titanic.py        # permanent
modal serve  modal_titanic.py        # ephemeral dev
"""

import modal
from pathlib import Path

app = modal.App("titanic-predictor")

LOCAL_DIR = Path(__file__).parent

# ── Build image: install deps AND bake app files in at build time ─
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
    .add_local_file(str(LOCAL_DIR / "titanic_app.py"),        "/app/titanic_app.py")
    .add_local_file(str(LOCAL_DIR / "titanic_svc_model.pkl"), "/app/titanic_svc_model.pkl")
    .add_local_file(str(LOCAL_DIR / "titanic.csv"),           "/app/titanic.csv")
)

# ── Web endpoint ──────────────────────────────────────────────────
@app.function(
    image=image,
    timeout=300,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8501, startup_timeout=60)
def run():
    import subprocess, sys
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "/app/titanic_app.py",
        "--server.port",               "8501",
        "--server.address",            "0.0.0.0",
        "--server.headless",           "true",
        "--server.enableCORS",         "false",
        "--server.enableXsrfProtection", "false",
    ])
