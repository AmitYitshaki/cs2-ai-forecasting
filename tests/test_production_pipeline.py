"""Integration checks for leakage-safe CS2 data preparation."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.data.splitter import ChronologicalSplitter
from src.features import MapDatasetPreparer, SymmetricFeatureEngineer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"


class ProductionPipelineTests(unittest.TestCase):
    """Verify filtering, ordering, and independent symmetrization."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_df = pd.read_csv(
            DATA_DIRECTORY / "final_tournament_features.csv", low_memory=False
        )
        cls.preparer = MapDatasetPreparer()
        cls.prepared_df, cls.audit = cls.preparer.transform(cls.raw_df)
        cls.base_splits = ChronologicalSplitter().split(cls.prepared_df)
        cls.engineer = SymmetricFeatureEngineer()

    def test_filter_removes_phantoms_and_invalid_scores(self) -> None:
        self.assertEqual(self.audit.phantom_rows, 1688)
        self.assertEqual(self.audit.non_positive_score_rows, 7)
        self.assertEqual(self.audit.valid_rows, 6700)
        self.assertFalse(self.prepared_df["map_name"].isna().any())
        self.assertNotIn("score1_game", self.prepared_df.columns)
        self.assertNotIn("score2_game", self.prepared_df.columns)

    def test_chronological_counts_before_symmetrization(self) -> None:
        expected_counts = {"train": 5472, "val": 633, "test": 595}
        actual_counts = {
            name: len(split_df) for name, split_df in self.base_splits.items()
        }
        self.assertEqual(actual_counts, expected_counts)

    def test_independent_feature_engineering(self) -> None:
        expected_counts = {"train": 10944, "val": 1266, "test": 1190}
        for name, split_df in self.base_splits.items():
            engineered_df = self.engineer.transform(split_df)
            self.assertEqual(len(engineered_df), expected_counts[name])
            self.assertEqual(len(engineered_df.columns), 33)
            self.assertAlmostEqual(engineered_df["team1_win"].mean(), 0.5)

if __name__ == "__main__":
    unittest.main()
