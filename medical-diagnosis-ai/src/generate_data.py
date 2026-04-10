

# src/generate_data.py

"""
Medical Diagnosis Decision Support System
Synthetic dataset generation with uncertainty
AIN7101 – Master Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """
    Synthetic medical diagnosis dataset:
    Features mix symptoms + tests with noise to simulate uncertainty.
    Labels: 4 classes: Flu, Pneumonia, Allergy, Bronchitis.
    """
    rng = np.random.default_rng(seed)

    # Symptoms (0-10 scales)
    fever = rng.normal(5.0, 2.0, n).clip(0, 10)
    cough = rng.normal(5.0, 2.5, n).clip(0, 10)
    sore_throat = rng.normal(4.0, 2.5, n).clip(0, 10)
    breath_short = rng.normal(3.5, 2.8, n).clip(0, 10)
    fatigue = rng.normal(5.0, 2.2, n).clip(0, 10)

    # Tests (uncertain/noisy)
    wbc = rng.normal(7.5, 2.0, n).clip(3, 15)         # (x10^9/L)
    crp = rng.normal(18, 15, n).clip(0, 120)           # mg/L
    spo2 = rng.normal(96, 2.0, n).clip(85, 100)        # %
    xray_infiltrate = rng.binomial(1, 0.25, n)         # uncertain binary

    # Latent scoring (soft labels) to assign diagnoses
    # Pneumonia: high cough + high breath_short + low SpO2 + infiltrate + high CRP/WBC
    pneu_score = (
        0.7 * cough + 1.0 * breath_short + 0.8 * (100 - spo2) / 5
        + 1.3 * xray_infiltrate + 0.6 * (crp / 20) + 0.5 * ((wbc - 7) / 2)
    )

    # Flu: fever + fatigue + sore throat, moderate cough, lower CRP, no infiltrate usually
    flu_score = (
        1.0 * fever + 0.9 * fatigue + 0.7 * sore_throat + 0.3 * cough
        - 0.3 * (crp / 20) - 0.6 * xray_infiltrate
    )

    # Allergy: sore throat mild, fever low, cough mild-moderate, CRP low, SpO2 ok
    allergy_score = (
        0.8 * (10 - fever) + 0.6 * (10 - crp / 20) + 0.4 * (10 - fatigue)
        + 0.2 * cough - 0.4 * xray_infiltrate
    )

    # Bronchitis: cough high, fever mild, breath_short mild-moderate, xray mostly negative
    bronch_score = (
        1.1 * cough + 0.4 * fever + 0.5 * breath_short + 0.2 * crp / 20
        - 0.7 * xray_infiltrate
    )

    scores = np.vstack([flu_score, pneu_score, allergy_score, bronch_score]).T
    probs = _sigmoid(scores - scores.mean(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    labels = np.array(["flu", "pneumonia", "allergy", "bronchitis"])
    y = labels[np.array([rng.choice(4, p=p) for p in probs])]

    df = pd.DataFrame(
        {
            "fever": fever,
            "cough": cough,
            "sore_throat": sore_throat,
            "breath_short": breath_short,
            "fatigue": fatigue,
            "wbc": wbc,
            "crp": crp,
            "spo2": spo2,
            "xray_infiltrate": xray_infiltrate,
            "diagnosis": y,
        }
    )

    # Add measurement uncertainty noise (simulate test uncertainty)
    df["wbc"] = (df["wbc"] + rng.normal(0, 0.4, n)).clip(3, 15)
    df["crp"] = (df["crp"] + rng.normal(0, 4.0, n)).clip(0, 120)
    df["spo2"] = (df["spo2"] + rng.normal(0, 0.7, n)).clip(85, 100)

    return df


if __name__ == "__main__":
    df = generate_synthetic()
    df.to_csv("data/synthetic_patients.csv", index=False)
    print("Saved: data/synthetic_patients.csv", df.shape)
