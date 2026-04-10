# src/ml_decision_tree.py

"""
Decision Tree-based Medical Diagnosis
Supervised machine learning approach
AIN7101 - Master Project
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


FEATURES_NUM = [
    "fever", "cough", "sore_throat", "breath_short", "fatigue", "wbc", "crp", "spo2"
]
FEATURES_CAT = ["xray_infiltrate"]
TARGET = "diagnosis"


@dataclass
class MLArtifacts:
    model: Pipeline
    feature_names: List[str]
    report: dict
    tree_text: str


def train_decision_tree(
    df: pd.DataFrame,
    seed: int = 7,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
) -> MLArtifacts:
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extract feature names after preprocessing
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(FEATURES_CAT))
    feature_names = FEATURES_NUM + cat_names

    tree = pipe.named_steps["clf"]
    tree_text = export_text(tree, feature_names=feature_names)

    return MLArtifacts(model=pipe, feature_names=feature_names, report=report, tree_text=tree_text)


def predict_with_proba(artifacts: MLArtifacts, x_row: pd.DataFrame) -> Tuple[str, dict]:
    proba = artifacts.model.predict_proba(x_row)[0]
    classes = artifacts.model.named_steps["clf"].classes_
    scores = {cls: float(p) for cls, p in zip(classes, proba)}
    label = max(scores, key=scores.get)
    return label, scores
