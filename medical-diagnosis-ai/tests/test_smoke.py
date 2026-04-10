import unittest
from pathlib import Path

import pandas as pd

from src.fuzzy_diagnosis import FuzzyDiagnosis, predict_label
from src.ml_decision_tree import FEATURES_CAT, FEATURES_NUM, train_decision_tree


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_patients.csv"


class TestProjectSmoke(unittest.TestCase):
    def test_dataset_exists(self) -> None:
        self.assertTrue(DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}")

    def test_fuzzy_inference_runs(self) -> None:
        df = pd.read_csv(DATA_PATH)
        row = df.iloc[0][FEATURES_NUM + FEATURES_CAT].to_dict()
        model = FuzzyDiagnosis()
        res = model.infer(row)
        label = predict_label(res.scores)
        self.assertIsInstance(label, str)
        self.assertTrue(len(res.scores) > 0)

    def test_decision_tree_train_and_predict(self) -> None:
        df = pd.read_csv(DATA_PATH)
        model = train_decision_tree(df, seed=7, max_depth=5, min_samples_leaf=10)
        X = df[FEATURES_NUM + FEATURES_CAT].head(3)
        preds = model.model.predict(X)
        self.assertEqual(len(preds), 3)


if __name__ == "__main__":
    unittest.main()
