"""Unit tests for the live-Elo series simulator."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.simulator import FrozenEloSnapshot, SeriesSimulator


class ConstantProbabilityModel:
    """Small deterministic model double for simulator tests."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features):
        positive = np.full(len(features), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class SeriesSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        snapshot = FrozenEloSnapshot(
            global_ratings={"a": 1500.0, "b": 1500.0},
            map_ratings={},
        )
        self.simulator = SeriesSimulator(
            calibrated_model=ConstantProbabilityModel(0.5),
            snapshot=snapshot,
            map_sequence=("M1", "M2", "M3", "M4", "M5"),
            random_state=7,
        )

    def test_bo5_distribution_is_complete_and_normalized(self) -> None:
        result = self.simulator.simulate("a", "b", best_of=5, n_iterations=1000)

        self.assertEqual(
            set(result.scoreline_distribution),
            {"3-0", "3-1", "3-2", "0-3", "1-3", "2-3"},
        )
        self.assertAlmostEqual(sum(result.scoreline_distribution.values()), 1.0)
        self.assertAlmostEqual(
            result.team_a_win_probability + result.team_b_win_probability, 1.0
        )

    def test_simulation_is_reproducible(self) -> None:
        first = self.simulator.simulate("a", "b", best_of=3, n_iterations=250)
        second = self.simulator.simulate("a", "b", best_of=3, n_iterations=250)

        self.assertEqual(first.scoreline_distribution, second.scoreline_distribution)

    def test_rejects_invalid_series(self) -> None:
        with self.assertRaises(ValueError):
            self.simulator.simulate("a", "b", best_of=7)
        with self.assertRaises(ValueError):
            self.simulator.simulate("a", "a", best_of=3)


if __name__ == "__main__":
    unittest.main()
