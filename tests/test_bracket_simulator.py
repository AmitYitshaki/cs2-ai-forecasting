"""Unit tests for single-elimination bracket simulation."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.bracket_simulator import (
    TournamentEloState,
    print_tournament_summary,
    simulate_double_elimination_tournament,
    simulate_series_once,
    simulate_tournament,
)


class ConstantProbabilityModel:
    def predict_proba(self, features):
        positive = np.full(len(features), 0.5, dtype=float)
        return np.column_stack((1.0 - positive, positive))


class BracketSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.teams = tuple("abcdefgh")
        self.maps = ("M1", "M2", "M3", "M4", "M5", "M6", "M7")
        self.state = TournamentEloState(
            global_ratings={team: 1500.0 for team in self.teams},
            map_ratings={
                (team, map_name): 1500.0
                for team in self.teams
                for map_name in self.maps
            },
        )
        self.model = ConstantProbabilityModel()

    def test_series_is_pure_with_respect_to_input_state(self) -> None:
        original_globals = self.state.global_ratings.copy()
        original_maps = self.state.map_ratings.copy()

        winner, updated, scoreline = simulate_series_once(
            "a",
            "b",
            self.state,
            3,
            self.maps[:3],
            calibrated_model=self.model,
            rng=np.random.default_rng(11),
        )

        self.assertIn(winner, {"a", "b"})
        self.assertIn(scoreline, {"2-0", "2-1", "0-2", "1-2"})
        self.assertIsNot(updated, self.state)
        self.assertEqual(self.state.global_ratings, original_globals)
        self.assertEqual(self.state.map_ratings, original_maps)

    def test_tournament_counts_and_summary_are_valid(self) -> None:
        quarterfinals = tuple(
            (self.teams[index], self.teams[index + 1])
            for index in range(0, 8, 2)
        )
        results = simulate_tournament(
            quarterfinals,
            self.state,
            self.model,
            self.maps,
            n_iterations=100,
            random_state=12,
            show_progress=False,
        )

        self.assertEqual(sum(row["Champion"] for row in results.values()), 100)
        for row in results.values():
            self.assertGreaterEqual(row["QF"], row["SF"])
            self.assertGreaterEqual(row["SF"], row["Final"])
            self.assertGreaterEqual(row["Final"], row["Champion"])

        summary = print_tournament_summary(results, 100)
        self.assertEqual(
            list(summary.columns),
            ["Team", "P(QF)", "P(SF)", "P(Final)", "P(Champion)", "SE(Champion)"],
        )
        self.assertEqual(len(summary), 8)

    def test_double_elimination_counts_are_valid_with_and_without_reset(self) -> None:
        opening_matchups = tuple(
            (self.teams[index], self.teams[index + 1])
            for index in range(0, 8, 2)
        )
        for reset in (False, True):
            with self.subTest(grand_final_reset=reset):
                results = simulate_double_elimination_tournament(
                    opening_matchups,
                    self.state,
                    self.model,
                    self.maps,
                    n_iterations=100,
                    random_state=19,
                    grand_final_reset=reset,
                    show_progress=False,
                )
                self.assertEqual(
                    sum(row["Champion"] for row in results.values()), 100
                )
                self.assertEqual(sum(row["SF"] for row in results.values()), 400)
                self.assertEqual(
                    sum(row["Final"] for row in results.values()), 200
                )
                for row in results.values():
                    self.assertGreaterEqual(row["QF"], row["SF"])
                    self.assertGreaterEqual(row["SF"], row["Final"])
                    self.assertGreaterEqual(row["Final"], row["Champion"])

    def test_double_elimination_analytics_reconcile(self) -> None:
        opening_matchups = tuple(
            (self.teams[index], self.teams[index + 1])
            for index in range(0, 8, 2)
        )
        results, analytics = simulate_double_elimination_tournament(
            opening_matchups,
            self.state,
            self.model,
            self.maps,
            n_iterations=200,
            random_state=23,
            track_analytics=True,
            show_progress=False,
        )
        self.assertEqual(sum(analytics.grand_final_matchups.values()), 200)
        self.assertEqual(sum(analytics.runner_ups.values()), 200)
        self.assertEqual(sum(analytics.exact_podiums.values()), 200)
        self.assertEqual(sum(analytics.grand_final_appearances.values()), 400)
        for team in self.teams:
            self.assertEqual(
                analytics.runner_ups[team] + results[team]["Champion"],
                results[team]["Final"],
            )


if __name__ == "__main__":
    unittest.main()
