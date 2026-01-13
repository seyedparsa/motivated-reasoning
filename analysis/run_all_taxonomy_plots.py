#!/usr/bin/env python3
"""
Generate all taxonomy plots (per-job, dataset aggregates, model aggregates, and overall aggregates).
"""
import subprocess
from pathlib import Path


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    base_cmd = ["python", "analysis/plot_taxonomy.py"]
    pdf_dir = Path("figures/taxonomy/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Individual plots for every model/dataset
    run(base_cmd + ["--all"])

    # Aggregates
    run(base_cmd + ["--aggregate"])
    run(base_cmd + ["--aggregate-by-dataset"])
    run(base_cmd + ["--aggregate-by-model"])

    # Optional: also save PDFs for the aggregate plots (store in dedicated folder)
    run(base_cmd + ["--aggregate", "--fmt", "pdf", "--save-dir", str(pdf_dir)])
    run(base_cmd + ["--aggregate-by-dataset", "--fmt", "pdf", "--save-dir", str(pdf_dir)])
    run(base_cmd + ["--aggregate-by-model", "--fmt", "pdf", "--save-dir", str(pdf_dir)])


if __name__ == "__main__":
    cwd = Path(__file__).resolve().parents[1]
    try:
        run(["python", "--version"])
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {exc}")

