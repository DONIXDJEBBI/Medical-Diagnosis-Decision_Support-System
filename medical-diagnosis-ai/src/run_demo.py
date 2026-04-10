# src/run_demo.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fuzzy_diagnosis import FuzzyDiagnosis, predict_label
from src.ml_decision_tree import FEATURES_CAT, FEATURES_NUM, train_decision_tree


def main():
    df = pd.read_csv(BASE_DIR / "data" / "synthetic_patients.csv")

    # Train ML model
    ml_art = train_decision_tree(df)

    # Pick one patient example
    patient = df.sample(1, random_state=3)[FEATURES_NUM + FEATURES_CAT]
    x = patient.iloc[0].to_dict()

    print("=== Patient Input ===")
    print(patient.to_string(index=False))

    # Fuzzy
    fuzzy = FuzzyDiagnosis()
    fz_res = fuzzy.infer(x)
    fz_label = predict_label(fz_res.scores)

    print("\n=== Fuzzy Output ===")
    print("Pred:", fz_label)
    print("Scores:", {k: round(v, 3) for k, v in fz_res.scores.items()})
    print("Top rules:")
    for name, strength in fz_res.fired_rules[:5]:
        print(f"- {name} -> {strength:.3f}")

    # ML
    ml_label = ml_art.model.predict(patient)[0]
    ml_proba = ml_art.model.predict_proba(patient)[0]
    classes = ml_art.model.named_steps["clf"].classes_
    scores = {c: float(p) for c, p in zip(classes, ml_proba)}

    print("\n=== Decision Tree Output ===")
    print("Pred:", ml_label)
    print("Prob:", {k: round(v, 3) for k, v in scores.items()})

    print("\n=== Tree (text) preview ===")
    print(ml_art.tree_text[:1200])


if __name__ == "__main__":
    main()
