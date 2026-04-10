#!/usr/bin/env python3
"""
Quick demo runner for the exam.
Runs comparison and prints a short summary for the instructor.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.compare import run_comparison


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_patients.csv"
RESULTS_PATH = BASE_DIR / "results" / "metrics.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("\n[DEMO] Running full comparison...")
    run_comparison()

    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle).get("effectiveness", {})
        print("\n[DEMO] Key results (from results/metrics.json):")
        for model, vals in metrics.items():
            acc = vals.get("accuracy")
            f1 = vals.get("f1_macro")
            print(f"  - {model}: accuracy={acc}, f1_macro={f1}")
    else:
        print("\n[DEMO] metrics.json not found yet. Run `python src/evaluate.py` if needed.")

    print("\n[DEMO] Note: This system uses synthetic data and is for education only.")


if __name__ == "__main__":
    main()
