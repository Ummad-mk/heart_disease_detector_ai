"""
scripts/generate_pdf_reports.py
===============================
Converts Jupyter Notebooks (.ipynb) into clean, high-resolution PDF reports.

Methodology:
1. Converts .ipynb -> HTML via nbconvert (preserves code, markdown, charts, tables).
2. Converts HTML -> PDF via headless Microsoft Edge print-to-pdf engine.
   (Avoids complex LaTeX/MiKTeX setup errors while preserving browser-quality styling).

Usage:
    python scripts/generate_pdf_reports.py
"""

from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser_exe() -> str:
    for path in EDGE_PATHS:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Could not find Microsoft Edge or Google Chrome executable.")


def convert_notebook_to_pdf(ipynb_path: Path, output_pdf_name: str, browser_exe: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = ipynb_path.stem
    html_path = REPORTS_DIR / f"{stem}.html"
    pdf_path = REPORTS_DIR / output_pdf_name

    print(f"[*] Converting {ipynb_path.name} -> HTML...")
    subprocess.run([
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        str(ipynb_path),
        "--output-dir", str(REPORTS_DIR)
    ], check=True)

    print(f"[*] Rendering {html_path.name} -> {output_pdf_name} via headless browser...")
    subprocess.run([
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        str(html_path)
    ], check=True)

    print(f"[+] Saved PDF report to: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


def main():
    browser = find_browser_exe()
    notebooks = [
        (NOTEBOOKS_DIR / "01_titanic_survival_prediction.ipynb", "Titanic_Survival_Prediction_Report.pdf"),
        (NOTEBOOKS_DIR / "02_heart_disease_prediction.ipynb", "Heart_Disease_Prediction_Report.pdf"),
    ]

    for nb_path, pdf_name in notebooks:
        if nb_path.exists():
            convert_notebook_to_pdf(nb_path, pdf_name, browser)
        else:
            print(f"[!] Warning: Notebook not found at {nb_path}")

    print("\n[SUCCESS] All PDF reports have been generated in outputs/reports/")


if __name__ == "__main__":
    main()
