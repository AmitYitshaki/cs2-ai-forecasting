from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


FEATURE_MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "features"
if str(FEATURE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_MODULE_DIR))

from player_elo_state import PlayerEloReplayEngine, V2_FEATURE_COLUMNS


class PlayerEloReplayEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        timestamps = [pd.Timestamp("2026-09-16 12:00:00"), pd.Timestamp("2026-09-17")]
        self.maps = pd.DataFrame(
            [
                {
                    "_row_id": index,
                    "datetime": timestamp,
                    "match_id": index + 1,
                    "game_id": index + 10,
                    "tournament": "fixture",
                    "map_name": "Mirage",
                    "team1": "Alpha",
                    "team2": "Beta",
                    "team1_join_key": "alpha",
                    "team2_join_key": "beta",
                    "team1_win": 1,
                }
                for index, timestamp in enumerate(timestamps)
            ]
        )
        player_rows = []
        for row_id, timestamp in enumerate(timestamps):
            for side in (1, 2):
                for slot in range(1, 6):
                    player_rows.append(
                        {
                            "_row_id": row_id,
                            "match_id": row_id + 1,
                            "datetime": timestamp,
                            "side": side,
                            "slot": slot,
                            "player_key": f"id:{side}{slot}",
                            "player_name": f"p{side}{slot}",
                            "player_norm": f"p{side}{slot}",
                            "relative_performance_basic": 0.0,
                            "relative_performance_rich": 0.0,
                            "has_rich_stats": False,
                        }
                    )
        self.players = pd.DataFrame(player_rows)

    def test_cutoff_is_exclusive_and_feature_order_is_locked(self) -> None:
        engine = PlayerEloReplayEngine(self.maps, self.players)
        cutoff = pd.Timestamp("2026-09-17T00:00:00")
        replay = engine.replay_until(cutoff)
        self.assertEqual(replay.replayed_rows, 1)
        self.assertLess(replay.last_replayed_timestamp, cutoff)
        self.assertTrue(all(timestamp < cutoff for timestamp in engine.replayed_timestamps_))

        feature = engine.build_matchup_feature_dict(
            "alpha",
            "beta",
            [f"id:1{slot}" for slot in range(1, 6)],
            [f"id:2{slot}" for slot in range(1, 6)],
            at=cutoff,
            player_scale_factor=2.0,
        )
        self.assertEqual(tuple(feature), V2_FEATURE_COLUMNS)
        self.assertEqual(feature["team1_player_cold_starts"], 0.0)
        self.assertEqual(feature["team2_player_cold_starts"], 0.0)

    def test_unseen_player_resolves_to_explicit_cold_start(self) -> None:
        engine = PlayerEloReplayEngine(self.maps, self.players)
        engine.replay_until(pd.Timestamp("2026-09-17T00:00:00"))
        key = engine.resolve_player("newplayer")
        self.assertEqual(key, "name:newplayer")
        self.assertTrue(engine.player_status(key)["cold_start"])


if __name__ == "__main__":
    unittest.main()
