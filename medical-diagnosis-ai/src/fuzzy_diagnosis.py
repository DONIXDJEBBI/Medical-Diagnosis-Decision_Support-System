# src/fuzzy_diagnosis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def tri(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


@dataclass
class FuzzyResult:
    scores: Dict[str, float]
    fired_rules: List[Tuple[str, float]]  # (rule_name, strength)


class FuzzyDiagnosis:
    """
    Fuzzy rule-based diagnosis for 4 classes:
    flu, pneumonia, allergy, bronchitis

    Inputs are normalized in realistic ranges:
    fever (0-10), cough (0-10), sore_throat (0-10), breath_short (0-10), fatigue (0-10),
    wbc (3-15), crp (0-120), spo2 (85-100), xray_infiltrate (0/1).
    """

    def fuzzify(self, x: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fever = x["fever"]
        cough = x["cough"]
        sore = x["sore_throat"]
        breath = x["breath_short"]
        fatigue = x["fatigue"]
        wbc = x["wbc"]
        crp = x["crp"]
        spo2 = x["spo2"]
        xray = x["xray_infiltrate"]

        fz = {
            "fever": {
                "low": trap(fever, 0, 0, 2.5, 4.5),
                "mid": tri(fever, 3, 5, 7),
                "high": trap(fever, 6, 7.5, 10, 10),
            },
            "cough": {
                "low": trap(cough, 0, 0, 2, 4),
                "mid": tri(cough, 3, 5, 7),
                "high": trap(cough, 6, 7.5, 10, 10),
            },
            "sore_throat": {
                "low": trap(sore, 0, 0, 2, 4),
                "high": trap(sore, 5, 6.5, 10, 10),
            },
            "breath_short": {
                "low": trap(breath, 0, 0, 2, 4),
                "high": trap(breath, 5, 6.5, 10, 10),
            },
            "fatigue": {
                "low": trap(fatigue, 0, 0, 2, 4),
                "high": trap(fatigue, 5, 6.5, 10, 10),
            },
            "wbc": {
                "normal": trap(wbc, 3, 5.5, 8.5, 10),
                "high": trap(wbc, 8.5, 10, 15, 15),
            },
            "crp": {
                "low": trap(crp, 0, 0, 10, 25),
                "high": trap(crp, 20, 35, 120, 120),
            },
            "spo2": {
                "normal": trap(spo2, 94, 96, 100, 100),
                "low": trap(spo2, 85, 85, 91, 94),
            },
            "xray_infiltrate": {
                "no": 1.0 if xray < 0.5 else 0.0,
                "yes": 1.0 if xray >= 0.5 else 0.0,
            },
        }
        return fz

    @staticmethod
    def _AND(*vals: float) -> float:
        return min(vals)

    @staticmethod
    def _OR(*vals: float) -> float:
        return max(vals)

    def infer(self, x: Dict[str, float]) -> FuzzyResult:
        fz = self.fuzzify(x)

        fired: List[Tuple[str, float]] = []
        score = {"flu": 0.0, "pneumonia": 0.0, "allergy": 0.0, "bronchitis": 0.0}

        # Rule set (interpretable)
        # Pneumonia rules
        r1 = self._AND(
            fz["cough"]["high"],
            fz["breath_short"]["high"],
            fz["spo2"]["low"],
            fz["crp"]["high"],
        )
        fired.append(("PNEU_R1: cough_high & breath_high & spo2_low & crp_high", r1))
        score["pneumonia"] = max(score["pneumonia"], r1)

        r2 = self._AND(
            fz["xray_infiltrate"]["yes"],
            self._OR(fz["wbc"]["high"], fz["crp"]["high"]),
            fz["cough"]["mid"],
        )
        fired.append(("PNEU_R2: xray_yes & (wbc_high OR crp_high) & cough_mid", r2))
        score["pneumonia"] = max(score["pneumonia"], r2)

        # Flu rules
        r3 = self._AND(fz["fever"]["high"], fz["fatigue"]["high"], fz["sore_throat"]["high"])
        fired.append(("FLU_R1: fever_high & fatigue_high & sore_high", r3))
        score["flu"] = max(score["flu"], r3)

        r4 = self._AND(fz["fever"]["mid"], fz["fatigue"]["high"], fz["crp"]["low"])
        fired.append(("FLU_R2: fever_mid & fatigue_high & crp_low", r4))
        score["flu"] = max(score["flu"], r4)

        # Allergy rules
        r5 = self._AND(fz["fever"]["low"], fz["crp"]["low"], fz["spo2"]["normal"])
        fired.append(("ALL_R1: fever_low & crp_low & spo2_normal", r5))
        score["allergy"] = max(score["allergy"], r5)

        r6 = self._AND(fz["sore_throat"]["low"], fz["fever"]["low"], fz["cough"]["low"])
        fired.append(("ALL_R2: sore_low & fever_low & cough_low", r6))
        score["allergy"] = max(score["allergy"], r6)

        # Bronchitis rules
        r7 = self._AND(fz["cough"]["high"], fz["xray_infiltrate"]["no"], fz["spo2"]["normal"])
        fired.append(("BRON_R1: cough_high & xray_no & spo2_normal", r7))
        score["bronchitis"] = max(score["bronchitis"], r7)

        r8 = self._AND(fz["cough"]["mid"], fz["fever"]["mid"], fz["breath_short"]["low"])
        fired.append(("BRON_R2: cough_mid & fever_mid & breath_low", r8))
        score["bronchitis"] = max(score["bronchitis"], r8)

        # Normalize scores (optional)
        s = sum(score.values())
        if s > 0:
            score = {k: v / s for k, v in score.items()}

        fired_sorted = sorted(fired, key=lambda t: t[1], reverse=True)
        return FuzzyResult(scores=score, fired_rules=fired_sorted)


def predict_label(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)
