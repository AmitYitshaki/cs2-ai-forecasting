"""Tests for point-in-time Elo and head-to-head features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from elo import PointInTimeEngine


class PointInTimeEngineTests(unittest.TestCase):
    """Protect the temporal ordering and current-match H2H boundary."""

    def test_h2h_is_strictly_prior_to_current_match(self) -> None:
        rows = pd.DataFrame(
            {
                "match_id": [20, 10, 10],
                "game_id": [3, 1, 2],
                "datetime": ["2024-01-02", "2024-01-01", "2024-01-01"],
                "map_name": ["Nuke", "Nuke", "Mirage"],
                "is_total": [False, False, False],
                "bestOf": [1, 3, 3],
                "score1_game": [13, 13, 8],
                "score2_game": [9, 7, 13],
                "team1_id": [101, 1, 1],
                "team2_id": [202, 2, 2],
                "team1_join_key": ["alpha", "alpha", "alpha"],
                "team2_join_key": ["beta", "beta", "beta"],
                "team1_win": [1, 1, 0],
            }
        )

        result = PointInTimeEngine().transform(rows)

        first_match = result[result["match_id"].eq(10)]
        next_match = result[result["match_id"].eq(20)].iloc[0]
        self.assertTrue(result["datetime"].is_monotonic_increasing)
        self.assertTrue(first_match["team1_h2h_wins"].eq(0).all())
        self.assertTrue(first_match["team2_h2h_wins"].eq(0).all())
        self.assertEqual(next_match["team1_h2h_wins"], 1)
        self.assertEqual(next_match["team2_h2h_wins"], 1)
        self.assertTrue(first_match["team1_roll5_win_pct"].eq(0).all())
        self.assertTrue(first_match["team1_roll5_round_diff"].eq(0).all())
        self.assertEqual(next_match["team1_roll5_win_pct"], 0.5)
        self.assertEqual(next_match["team2_roll5_win_pct"], 0.5)
        self.assertEqual(next_match["team1_roll5_round_diff"], 1.0)
        self.assertEqual(next_match["team2_roll5_round_diff"], -1.0)


if __name__ == "__main__":
    unittest.main()
