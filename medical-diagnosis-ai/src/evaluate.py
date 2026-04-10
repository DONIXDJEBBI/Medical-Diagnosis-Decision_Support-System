"""
Offline evaluation of Fuzzy Logic vs Decision Tree.
Metrics: accuracy, F1-score (macro), confusion matrix, ROC/AUC.
AIN7101 Master Project.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fuzzy_diagnosis import FuzzyDiagnosis, predict_label
from src.ml_decision_tree import FEATURES_CAT, FEATURES_NUM, train_decision_tree

DATA_PATH = BASE_DIR / "data" / "synthetic_patients.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "metrics.json"

POSSIBLE_LABELS = ["label", "diagnosis", "disease", "target", "class"]


def _compute_effectiveness(y_true: pd.Series, y_pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def _per_class_f1(y_true: pd.Series, y_pred: List[str]) -> Dict[str, float]:
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    classes = sorted(y_true.unique().tolist())
    return {
        cls: round(report[cls]["f1-score"], 4)
        for cls in classes
        if cls in report
    }


def _summarize_cv(metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    keys = metrics[0].keys() if metrics else []
    for key in keys:
        vals = np.array([m[key] for m in metrics], dtype=float)
        summary[key] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=1)), 4),
        }
    return summary


def _summarize_per_class(per_class_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not per_class_list:
        return summary
    classes = sorted(per_class_list[0].keys())
    for cls in classes:
        vals = np.array([d.get(cls, 0.0) for d in per_class_list], dtype=float)
        summary[cls] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=1)), 4),
        }
    return summary


def _load_dataset() -> Tuple[pd.DataFrame, str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    label_col = next((col for col in POSSIBLE_LABELS if col in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"No label column found. Tried: {POSSIBLE_LABELS}. "
            f"Available columns: {list(df.columns)}"
        )
    return df, label_col


def _compute_roc(
    y_true_bin: np.ndarray,
    probas: np.ndarray,
    classes: List[str],
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, float], float]:
    per_class = {}
    aucs = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probas[:, i])
        per_class[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        aucs[cls] = float(roc_auc_score(y_true_bin[:, i], probas[:, i]))

    macro_auc = float(
        roc_auc_score(y_true_bin, probas, average="macro", multi_class="ovr")
    )
    return per_class, aucs, macro_auc


def run_evaluation(
    model_type: Literal["fuzzy_only", "tree_only", "both"] = "both",
    cv_folds: int = 5,
    seed: int = 7,
    tree_max_depth: int = 5,
    tree_min_samples_leaf: int = 10,
) -> None:
    """
    Run evaluation and save metrics to results/metrics.json.

    Args:
        model_type: Which model(s) to evaluate ("fuzzy_only", "tree_only", or "both").
        tree_max_depth: Decision tree max depth (default: 5).
        tree_min_samples_leaf: Decision tree min samples per leaf (default: 10).
    """
    df, label_col = _load_dataset()
    y_true = df[label_col]
    X = df[FEATURES_NUM + FEATURES_CAT]

    classes = sorted(y_true.unique().tolist())
    y_true_bin = label_binarize(y_true, classes=classes)

    do_fuzzy = model_type in ["fuzzy_only", "both"]
    do_tree = model_type in ["tree_only", "both"]

    fuzzy = FuzzyDiagnosis() if do_fuzzy else None
    ml = (
        train_decision_tree(
            df,
            seed=seed,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
        )
        if do_tree
        else None
    )

    fz_preds: List[str] = []
    ml_preds: List[str] = []
    fz_proba: List[List[float]] = []
    ml_proba: List[List[float]] = []

    ml_class_order = None
    if do_tree:
        ml_class_order = list(ml.model.named_steps["clf"].classes_)

    for _, row in X.iterrows():
        patient = row.to_dict()

        if do_fuzzy and fuzzy is not None:
            fz_res = fuzzy.infer(patient)
            fz_pred = predict_label(fz_res.scores)
            fz_preds.append(fz_pred)
            fz_proba.append([float(fz_res.scores.get(cls, 0.0)) for cls in classes])

        if do_tree and ml is not None:
            row_df = pd.DataFrame([row])
            ml_pred = ml.model.predict(row_df)[0]
            ml_preds.append(ml_pred)
            ml_probs = ml.model.predict_proba(row_df)[0]
            ml_scores = {cls: float(p) for cls, p in zip(ml_class_order, ml_probs)}
            ml_proba.append([ml_scores.get(cls, 0.0) for cls in classes])

    results: Dict[str, Dict] = {"effectiveness": {}, "diagnostics": {}, "metadata": {}}

    if do_fuzzy:
        results["effectiveness"]["fuzzy_logic"] = _compute_effectiveness(
            y_true, fz_preds
        )
        results["effectiveness"]["fuzzy_logic"]["per_class_f1"] = _per_class_f1(
            y_true, fz_preds
        )

    if do_tree:
        results["effectiveness"]["decision_tree"] = _compute_effectiveness(
            y_true, ml_preds
        )
        results["effectiveness"]["decision_tree"]["per_class_f1"] = _per_class_f1(
            y_true, ml_preds
        )

    results["diagnostics"]["classes"] = classes
    results["diagnostics"]["confusion_matrix"] = {}
    results["diagnostics"]["roc"] = {}

    if do_fuzzy:
        cm_fz = confusion_matrix(y_true, fz_preds, labels=classes)
        results["diagnostics"]["confusion_matrix"]["fuzzy_logic"] = cm_fz.tolist()

        fz_proba_arr = np.array(fz_proba)
        fz_per_class, fz_auc, fz_macro_auc = _compute_roc(
            y_true_bin, fz_proba_arr, classes
        )
        results["diagnostics"]["roc"]["fuzzy_logic"] = {
            "per_class": fz_per_class,
            "auc": fz_auc,
            "macro_auc": round(fz_macro_auc, 4),
        }

    if do_tree:
        cm_ml = confusion_matrix(y_true, ml_preds, labels=classes)
        results["diagnostics"]["confusion_matrix"]["decision_tree"] = cm_ml.tolist()

        ml_proba_arr = np.array(ml_proba)
        ml_per_class, ml_auc, ml_macro_auc = _compute_roc(
            y_true_bin, ml_proba_arr, classes
        )
        results["diagnostics"]["roc"]["decision_tree"] = {
            "per_class": ml_per_class,
            "auc": ml_auc,
            "macro_auc": round(ml_macro_auc, 4),
        }

    results["metadata"] = {
        "dataset": str(DATA_PATH),
        "label_column": label_col,
        "num_samples": int(len(df)),
    }

    if cv_folds > 1:
        results["cross_validation"] = {"folds": cv_folds}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        if do_fuzzy:
            fz_metrics = []
            fz_per_class = []
            fuzzy_cv = FuzzyDiagnosis()
            for _, test_idx in skf.split(X, y_true):
                X_test = X.iloc[test_idx]
                y_test = y_true.iloc[test_idx]
                preds = []
                for _, row in X_test.iterrows():
                    res = fuzzy_cv.infer(row.to_dict())
                    preds.append(predict_label(res.scores))
                fz_metrics.append(_compute_effectiveness(y_test, preds))
                fz_per_class.append(_per_class_f1(y_test, preds))
            results["cross_validation"]["fuzzy_logic"] = {
                "metrics": _summarize_cv(fz_metrics),
                "per_class_f1": _summarize_per_class(fz_per_class),
            }

        if do_tree:
            ml_metrics = []
            ml_per_class = []
            for train_idx, test_idx in skf.split(X, y_true):
                df_train = df.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y_true.iloc[test_idx]
                ml_cv = train_decision_tree(
                    df_train,
                    seed=seed,
                    max_depth=tree_max_depth,
                    min_samples_leaf=tree_min_samples_leaf,
                )
                preds = ml_cv.model.predict(X_test)
                ml_metrics.append(_compute_effectiveness(y_test, preds.tolist()))
                ml_per_class.append(_per_class_f1(y_test, preds.tolist()))
            results["cross_validation"]["decision_tree"] = {
                "metrics": _summarize_cv(ml_metrics),
                "per_class_f1": _summarize_per_class(ml_per_class),
            }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("[OK] Evaluation completed successfully.")
    print(f"[RESULTS] Results saved to: {RESULTS_PATH}")
    print("[SUMMARY] Summary:")
    print(json.dumps(results["effectiveness"], indent=2))


if __name__ == "__main__":
    run_evaluation()
