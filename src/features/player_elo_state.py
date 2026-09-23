"""Leakage-safe point-in-time Team/Player Elo state replay for Version 2.0."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MU = 1500.0
K_TEAM = 24.0
TEAM_HALF_LIFE_DAYS = 180.0
PLAYER_HALF_LIFE_DAYS = 1095.0
K_PERF = 0.0

V2_FEATURE_COLUMNS = (
    "BaseElo_diff",
    "PlayerAggElo_diff_scaled",
    "team1_gap_days",
    "team2_gap_days",
    "player_elo_std_A_scaled",
    "player_elo_std_B_scaled",
    "lineup_prior_maps_diff",
    "team1_player_cold_starts",
    "team2_player_cold_starts",
)


def elo_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def decay_rating(
    stored_rating: float,
    last_seen: pd.Timestamp | None,
    now: pd.Timestamp,
    half_life_days: float,
    *,
    initial_rating: float = MU,
) -> tuple[float, float]:
    if last_seen is None:
        return float(initial_rating), 0.0
    gap_days = max((now - last_seen).total_seconds() / 86_400.0, 0.0)
    effective = initial_rating + (stored_rating - initial_rating) * 2.0 ** (
        -gap_days / half_life_days
    )
    return float(effective), float(gap_days)


def rating_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def build_v2_model_feature_frame(
    frame: pd.DataFrame,
    *,
    player_scale_factor: float,
) -> pd.DataFrame:
    """Build the locked nine-feature matrix by semantic column name."""

    output = pd.DataFrame(index=frame.index)
    output["BaseElo_diff"] = (
        frame["team1_elo_decay"].to_numpy(dtype=float)
        - frame["team2_elo_decay"].to_numpy(dtype=float)
    )
    output["PlayerAggElo_diff_scaled"] = player_scale_factor * (
        frame["team1_player_agg_elo"].to_numpy(dtype=float)
        - frame["team2_player_agg_elo"].to_numpy(dtype=float)
    )
    output["team1_gap_days"] = frame["team1_gap_days"].to_numpy(dtype=float)
    output["team2_gap_days"] = frame["team2_gap_days"].to_numpy(dtype=float)
    output["player_elo_std_A_scaled"] = player_scale_factor * frame[
        "team1_player_elo_decay_std"
    ].to_numpy(dtype=float)
    output["player_elo_std_B_scaled"] = player_scale_factor * frame[
        "team2_player_elo_decay_std"
    ].to_numpy(dtype=float)
    output["lineup_prior_maps_diff"] = frame["lineup_prior_maps_diff"].to_numpy(
        dtype=float
    )
    output["team1_player_cold_starts"] = frame[
        "team1_player_cold_starts"
    ].to_numpy(dtype=float)
    output["team2_player_cold_starts"] = frame[
        "team2_player_cold_starts"
    ].to_numpy(dtype=float)
    return output.loc[:, list(V2_FEATURE_COLUMNS)]


def symmetrize_v2_features(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Mirror a feature matrix without crossing a chronological boundary."""

    mirrored = features.copy()
    mirrored["BaseElo_diff"] = -features["BaseElo_diff"].to_numpy()
    mirrored["PlayerAggElo_diff_scaled"] = -features[
        "PlayerAggElo_diff_scaled"
    ].to_numpy()
    mirrored["lineup_prior_maps_diff"] = -features[
        "lineup_prior_maps_diff"
    ].to_numpy()
    for left, right in (
        ("team1_gap_days", "team2_gap_days"),
        ("player_elo_std_A_scaled", "player_elo_std_B_scaled"),
        ("team1_player_cold_starts", "team2_player_cold_starts"),
    ):
        mirrored[left] = features[right].to_numpy()
        mirrored[right] = features[left].to_numpy()
    return (
        pd.concat([features, mirrored], ignore_index=True),
        pd.concat([target.astype(int), 1 - target.astype(int)], ignore_index=True),
    )


@dataclass(frozen=True)
class ReplayResult:
    as_of_date: pd.Timestamp
    features: pd.DataFrame
    replayed_rows: int
    last_replayed_timestamp: pd.Timestamp | None


def _matchup_feature_dict(
    team1_key: str,
    team2_key: str,
    team1_players: Sequence[str],
    team2_players: Sequence[str],
    *,
    at: pd.Timestamp,
    player_scale_factor: float,
    team_ratings: Mapping[str, float],
    team_last_seen: Mapping[str, pd.Timestamp],
    player_ratings: Mapping[str, float],
    player_last_seen: Mapping[str, pd.Timestamp],
    player_map_counts: Mapping[str, int],
    lineup_map_counts: Mapping[tuple[str, ...], int],
    team_half_life_days: float,
    player_half_life_days: float,
    initial_rating: float,
) -> dict[str, float]:
    """Single source of truth for the locked nine-feature matchup contract."""

    if len(team1_players) != 5 or len(team2_players) != 5:
        raise ValueError("Each matchup side must contain exactly five players.")
    now = pd.Timestamp(at)
    team_values: dict[int, float] = {}
    team_gaps: dict[int, float] = {}
    player_values: dict[int, np.ndarray] = {}
    lineups = {
        1: tuple(sorted(map(str, team1_players))),
        2: tuple(sorted(map(str, team2_players))),
    }
    for side, (team_key, roster) in enumerate(
        ((team1_key, team1_players), (team2_key, team2_players)), start=1
    ):
        team_values[side], team_gaps[side] = decay_rating(
            team_ratings.get(team_key, initial_rating),
            team_last_seen.get(team_key),
            now,
            team_half_life_days,
            initial_rating=initial_rating,
        )
        player_values[side] = np.asarray(
            [
                decay_rating(
                    player_ratings.get(key, initial_rating),
                    player_last_seen.get(key),
                    now,
                    player_half_life_days,
                    initial_rating=initial_rating,
                )[0]
                for key in roster
            ],
            dtype=float,
        )

    feature = {
        "BaseElo_diff": team_values[1] - team_values[2],
        "PlayerAggElo_diff_scaled": player_scale_factor
        * (player_values[1].mean() - player_values[2].mean()),
        "team1_gap_days": team_gaps[1],
        "team2_gap_days": team_gaps[2],
        "player_elo_std_A_scaled": player_scale_factor
        * player_values[1].std(ddof=0),
        "player_elo_std_B_scaled": player_scale_factor
        * player_values[2].std(ddof=0),
        "lineup_prior_maps_diff": float(
            lineup_map_counts.get(lineups[1], 0)
            - lineup_map_counts.get(lineups[2], 0)
        ),
        "team1_player_cold_starts": float(
            sum(player_map_counts.get(key, 0) == 0 for key in team1_players)
        ),
        "team2_player_cold_starts": float(
            sum(player_map_counts.get(key, 0) == 0 for key in team2_players)
        ),
    }
    if tuple(feature) != V2_FEATURE_COLUMNS:
        raise AssertionError("V2 feature mapping order drifted from the locked contract.")
    return {name: float(feature[name]) for name in V2_FEATURE_COLUMNS}


@dataclass
class PlayerEloTournamentState:
    """Compact, independently mutable V2 state for one tournament path."""

    team_ratings: dict[str, float]
    team_last_seen: dict[str, pd.Timestamp]
    player_ratings: dict[str, float]
    player_last_seen: dict[str, pd.Timestamp]
    player_map_counts: dict[str, int]
    lineup_map_counts: dict[tuple[str, ...], int]
    rosters: Mapping[str, tuple[str, ...]]
    at: pd.Timestamp
    player_scale_factor: float
    initial_rating: float = MU
    k_team: float = K_TEAM
    team_half_life_days: float = TEAM_HALF_LIFE_DAYS
    player_half_life_days: float = PLAYER_HALF_LIFE_DAYS

    @classmethod
    def from_replay_engine(
        cls,
        engine: "PlayerEloReplayEngine",
        *,
        resolved_teams: Mapping[str, str],
        resolved_rosters: Mapping[str, Sequence[str]],
        at: pd.Timestamp,
        player_scale_factor: float,
    ) -> "PlayerEloTournamentState":
        rosters = {
            display: tuple(map(str, resolved_rosters[display]))
            for display in resolved_teams
        }
        players = {player for roster in rosters.values() for player in roster}
        lineup_keys = {tuple(sorted(roster)) for roster in rosters.values()}
        return cls(
            team_ratings={
                display: float(
                    engine.team_ratings.get(canonical, engine.initial_rating)
                )
                for display, canonical in resolved_teams.items()
            },
            team_last_seen={
                display: engine.team_last_seen[canonical]
                for display, canonical in resolved_teams.items()
                if canonical in engine.team_last_seen
            },
            player_ratings={
                player: float(
                    engine.player_ratings_decay.get(player, engine.initial_rating)
                )
                for player in players
            },
            player_last_seen={
                player: engine.player_last_seen[player]
                for player in players
                if player in engine.player_last_seen
            },
            player_map_counts={
                player: int(engine.player_map_counts[player]) for player in players
            },
            lineup_map_counts={
                lineup: int(engine.lineup_map_counts[lineup])
                for lineup in lineup_keys
            },
            rosters=rosters,
            at=pd.Timestamp(at),
            player_scale_factor=float(player_scale_factor),
            initial_rating=engine.initial_rating,
            k_team=engine.k_team,
            team_half_life_days=engine.team_half_life_days,
            player_half_life_days=engine.player_half_life_days,
        )

    def __deepcopy__(self, memo: dict) -> "PlayerEloTournamentState":
        clone = type(self)(
            team_ratings=self.team_ratings.copy(),
            team_last_seen=self.team_last_seen.copy(),
            player_ratings=self.player_ratings.copy(),
            player_last_seen=self.player_last_seen.copy(),
            player_map_counts=self.player_map_counts.copy(),
            lineup_map_counts=self.lineup_map_counts.copy(),
            rosters=self.rosters,
            at=self.at,
            player_scale_factor=self.player_scale_factor,
            initial_rating=self.initial_rating,
            k_team=self.k_team,
            team_half_life_days=self.team_half_life_days,
            player_half_life_days=self.player_half_life_days,
        )
        memo[id(self)] = clone
        return clone

    def build_matchup_feature_dict(self, team1: str, team2: str) -> dict[str, float]:
        return _matchup_feature_dict(
            team1,
            team2,
            self.rosters[team1],
            self.rosters[team2],
            at=self.at,
            player_scale_factor=self.player_scale_factor,
            team_ratings=self.team_ratings,
            team_last_seen=self.team_last_seen,
            player_ratings=self.player_ratings,
            player_last_seen=self.player_last_seen,
            player_map_counts=self.player_map_counts,
            lineup_map_counts=self.lineup_map_counts,
            team_half_life_days=self.team_half_life_days,
            player_half_life_days=self.player_half_life_days,
            initial_rating=self.initial_rating,
        )

    def update_map(self, team1: str, team2: str, team1_won: bool) -> None:
        effective: dict[str, float] = {}
        for team in (team1, team2):
            effective[team] = decay_rating(
                self.team_ratings.get(team, self.initial_rating),
                self.team_last_seen.get(team),
                self.at,
                self.team_half_life_days,
                initial_rating=self.initial_rating,
            )[0]
        actual = float(team1_won)
        delta = self.k_team * (
            actual - elo_probability(effective[team1], effective[team2])
        )
        for team, team_delta in ((team1, delta), (team2, -delta)):
            self.team_ratings[team] = effective[team] + team_delta
            self.team_last_seen[team] = self.at
            roster = self.rosters[team]
            for player in roster:
                player_rating = decay_rating(
                    self.player_ratings.get(player, self.initial_rating),
                    self.player_last_seen.get(player),
                    self.at,
                    self.player_half_life_days,
                    initial_rating=self.initial_rating,
                )[0]
                self.player_ratings[player] = player_rating + team_delta / 5.0
                self.player_last_seen[player] = self.at
                self.player_map_counts[player] = self.player_map_counts.get(player, 0) + 1
            lineup = tuple(sorted(roster))
            self.lineup_map_counts[lineup] = self.lineup_map_counts.get(lineup, 0) + 1


class PlayerEloReplayEngine:
    """Replay all completed maps strictly before an exclusive timestamp cutoff."""

    def __init__(
        self,
        maps: pd.DataFrame,
        players: pd.DataFrame,
        *,
        initial_rating: float = MU,
        k_team: float = K_TEAM,
        team_half_life_days: float = TEAM_HALF_LIFE_DAYS,
        player_half_life_days: float = PLAYER_HALF_LIFE_DAYS,
        k_perf: float = K_PERF,
        team_normalizer: Callable[[object], str] | None = None,
        player_normalizer: Callable[[object], str] | None = None,
    ) -> None:
        self.maps = maps.copy()
        self.players = players.copy()
        self.initial_rating = float(initial_rating)
        self.k_team = float(k_team)
        self.team_half_life_days = float(team_half_life_days)
        self.player_half_life_days = float(player_half_life_days)
        self.k_perf = float(k_perf)
        self.team_normalizer = team_normalizer
        self.player_normalizer = player_normalizer

        required_maps = {
            "_row_id", "datetime", "match_id", "game_id", "team1_win",
            "team1", "team2", "team1_join_key", "team2_join_key",
        }
        required_players = {
            "_row_id", "side", "slot", "player_key",
            "relative_performance_basic", "relative_performance_rich",
            "has_rich_stats",
        }
        missing_maps = required_maps - set(self.maps)
        missing_players = required_players - set(self.players)
        if missing_maps or missing_players:
            raise ValueError(
                f"Replay inputs are incomplete: maps={sorted(missing_maps)}, "
                f"players={sorted(missing_players)}"
            )
        self.maps["datetime"] = pd.to_datetime(self.maps["datetime"], errors="raise")
        self.players["datetime"] = pd.to_datetime(
            self.players["datetime"], errors="raise"
        )
        self.maps = self.maps.sort_values(
            ["datetime", "match_id", "game_id"], kind="stable"
        ).reset_index(drop=True)
        self.players = self.players.sort_values(
            ["_row_id", "side", "slot"], kind="stable"
        ).reset_index(drop=True)
        player_sizes = self.players.groupby(["_row_id", "side"], sort=False).size()
        if not player_sizes.eq(5).all():
            raise ValueError("Every map side must contain exactly five players.")
        self.player_groups = {
            (int(row_id), int(side)): group.reset_index(drop=True)
            for (row_id, side), group in self.players.groupby(
                ["_row_id", "side"], sort=False
            )
        }
        self._reset_state()

    def _reset_state(self) -> None:
        self.team_ratings: dict[str, float] = {}
        self.team_last_seen: dict[str, pd.Timestamp] = {}
        self.team_map_counts: defaultdict[str, int] = defaultdict(int)
        self.player_ratings_basic: dict[str, float] = {}
        self.player_ratings_rich: dict[str, float] = {}
        self.player_ratings_decay: dict[str, float] = {}
        self.player_last_seen: dict[str, pd.Timestamp] = {}
        self.player_map_counts: defaultdict[str, int] = defaultdict(int)
        self.player_rich_update_counts: defaultdict[str, int] = defaultdict(int)
        self.lineup_map_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
        self.team_aliases: defaultdict[str, set[str]] = defaultdict(set)
        self.player_aliases: defaultdict[str, set[str]] = defaultdict(set)
        self.feature_frame_ = pd.DataFrame()
        self.as_of_date_: pd.Timestamp | None = None
        self.replayed_timestamps_: list[pd.Timestamp] = []

    @staticmethod
    def _team_key(row: Mapping[str, object], side: int) -> str:
        canonical = row.get(f"team{side}_join_key")
        if pd.notna(canonical) and str(canonical).strip():
            return str(canonical).strip().casefold()
        fallback = row.get(f"team{side}_id", row[f"team{side}"])
        return str(fallback).strip().casefold()

    def replay_until(self, as_of_date: pd.Timestamp) -> ReplayResult:
        cutoff = pd.Timestamp(as_of_date)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)
        self._reset_state()
        history = self.maps.loc[self.maps["datetime"].lt(cutoff)]
        if history.empty:
            self.as_of_date_ = cutoff
            return ReplayResult(cutoff, pd.DataFrame(), 0, None)
        if not history["datetime"].lt(cutoff).all():
            raise AssertionError("Replay cutoff is exclusive.")

        feature_rows: list[dict[str, object]] = []
        for row in history.to_dict(orient="records"):
            now = pd.Timestamp(row["datetime"])
            row_id = int(row["_row_id"])
            team_keys = {side: self._team_key(row, side) for side in (1, 2)}
            for side in (1, 2):
                if self.team_normalizer is not None:
                    alias = self.team_normalizer(row[f"team{side}"])
                    if alias:
                        self.team_aliases[alias].add(team_keys[side])

            effective_team: dict[int, float] = {}
            team_gap: dict[int, float] = {}
            rosters: dict[int, pd.DataFrame] = {}
            lineup_keys: dict[int, tuple[str, ...]] = {}
            player_values: dict[str, dict[int, np.ndarray]] = {
                "basic": {}, "rich": {}, "decay": {}
            }
            roster_features: dict[int, dict[str, float | int | bool]] = {}

            for side in (1, 2):
                team_key = team_keys[side]
                team_rating, gap = decay_rating(
                    self.team_ratings.get(team_key, self.initial_rating),
                    self.team_last_seen.get(team_key),
                    now,
                    self.team_half_life_days,
                    initial_rating=self.initial_rating,
                )
                self.team_ratings[team_key] = team_rating
                effective_team[side] = team_rating
                team_gap[side] = gap

                roster = self.player_groups[(row_id, side)]
                rosters[side] = roster
                basic_values: list[float] = []
                rich_values: list[float] = []
                decay_values: list[float] = []
                gaps: list[float] = []
                prior_counts: list[int] = []
                prior_rich_counts: list[int] = []
                for player_row in roster.itertuples(index=False):
                    player_key = str(player_row.player_key)
                    player_norm = getattr(player_row, "player_norm", "")
                    player_name = getattr(player_row, "player_name", "")
                    if not player_norm and self.player_normalizer is not None:
                        player_norm = self.player_normalizer(player_name)
                    if player_norm:
                        self.player_aliases[str(player_norm)].add(player_key)
                    basic_values.append(
                        self.player_ratings_basic.get(player_key, self.initial_rating)
                    )
                    rich_values.append(
                        self.player_ratings_rich.get(player_key, self.initial_rating)
                    )
                    decayed_rating, player_gap = decay_rating(
                        self.player_ratings_decay.get(player_key, self.initial_rating),
                        self.player_last_seen.get(player_key),
                        now,
                        self.player_half_life_days,
                        initial_rating=self.initial_rating,
                    )
                    self.player_ratings_decay[player_key] = decayed_rating
                    decay_values.append(decayed_rating)
                    gaps.append(player_gap)
                    prior_counts.append(self.player_map_counts[player_key])
                    prior_rich_counts.append(self.player_rich_update_counts[player_key])

                for variant, values in (
                    ("basic", basic_values), ("rich", rich_values), ("decay", decay_values)
                ):
                    player_values[variant][side] = np.asarray(values, dtype=float)
                lineup_key = tuple(sorted(roster["player_key"].astype(str).tolist()))
                lineup_keys[side] = lineup_key
                current_rich_count = int(roster["has_rich_stats"].sum())
                roster_features[side] = {
                    "player_prior_maps_mean": float(np.mean(prior_counts)),
                    "player_cold_starts": int(np.sum(np.asarray(prior_counts) == 0)),
                    "lineup_prior_maps": int(self.lineup_map_counts[lineup_key]),
                    "player_gap_days_mean": float(np.mean(gaps)),
                    "player_gap_days_max": float(np.max(gaps)),
                    "player_prior_rich_updates_mean": float(np.mean(prior_rich_counts)),
                    "has_prior_rich_stats": bool(np.sum(prior_rich_counts) > 0),
                    "rich_stats_final_map": bool(current_rich_count > 0),
                    "rich_players_final_map": current_rich_count,
                }
                for variant in ("basic", "rich", "decay"):
                    for statistic, value in rating_summary(
                        player_values[variant][side]
                    ).items():
                        roster_features[side][f"player_elo_{variant}_{statistic}"] = value

            expected_team1 = elo_probability(effective_team[1], effective_team[2])
            feature: dict[str, object] = {
                "_row_id": row_id,
                "datetime": now,
                "match_id": row["match_id"],
                "game_id": row["game_id"],
                "tournament": row.get("tournament"),
                "map_name": row.get("map_name"),
                "team1": row["team1"],
                "team2": row["team2"],
                "team1_join_key": team_keys[1],
                "team2_join_key": team_keys[2],
                "team1_win": int(row["team1_win"]),
                "team1_elo_decay": effective_team[1],
                "team2_elo_decay": effective_team[2],
                "team1_gap_days": team_gap[1],
                "team2_gap_days": team_gap[2],
                "raw_team_elo_prob": expected_team1,
            }
            for side in (1, 2):
                for name, value in roster_features[side].items():
                    feature[f"team{side}_{name}"] = value
            feature["team_elo_diff"] = effective_team[1] - effective_team[2]
            for variant in ("basic", "rich", "decay"):
                probability = elo_probability(
                    float(player_values[variant][1].mean()),
                    float(player_values[variant][2].mean()),
                )
                feature[f"raw_player_elo_prob_{variant}"] = probability
                feature[f"player_elo_{variant}_mean_diff"] = (
                    roster_features[1][f"player_elo_{variant}_mean"]
                    - roster_features[2][f"player_elo_{variant}_mean"]
                )
                feature[f"player_elo_{variant}_min_diff"] = (
                    roster_features[1][f"player_elo_{variant}_min"]
                    - roster_features[2][f"player_elo_{variant}_min"]
                )
            feature["raw_player_elo_prob"] = feature["raw_player_elo_prob_decay"]
            feature["lineup_prior_maps_diff"] = (
                roster_features[1]["lineup_prior_maps"]
                - roster_features[2]["lineup_prior_maps"]
            )
            for side in (1, 2):
                for statistic in ("mean", "median", "min", "max", "std"):
                    feature[f"team{side}_player_elo_{statistic}"] = roster_features[
                        side
                    ][f"player_elo_decay_{statistic}"]
            feature["player_elo_mean_diff"] = feature["player_elo_decay_mean_diff"]
            feature["player_elo_min_diff"] = feature["player_elo_decay_min_diff"]
            feature["team1_player_agg_elo"] = feature[
                "team1_player_elo_decay_mean"
            ]
            feature["team2_player_agg_elo"] = feature[
                "team2_player_elo_decay_mean"
            ]
            feature["player_agg_elo_diff"] = (
                feature["team1_player_agg_elo"]
                - feature["team2_player_agg_elo"]
            )
            feature_rows.append(feature)

            actual = float(row["team1_win"])
            team1_delta = self.k_team * (actual - expected_team1)
            team_deltas = {1: team1_delta, 2: -team1_delta}
            for side in (1, 2):
                team_key = team_keys[side]
                self.team_ratings[team_key] = effective_team[side] + team_deltas[side]
                self.team_last_seen[team_key] = now
                self.team_map_counts[team_key] += 1
                roster = rosters[side]
                basic_relative = roster[
                    "relative_performance_basic"
                ].to_numpy(dtype=float)
                rich_relative = roster[
                    "relative_performance_rich"
                ].to_numpy(dtype=float)
                if not np.isclose(basic_relative.sum(), 0.0, atol=1e-10):
                    raise AssertionError("Basic relative performance is not centered.")
                if not np.isclose(rich_relative.sum(), 0.0, atol=1e-10):
                    raise AssertionError("Rich relative performance is not centered.")
                basic_deltas = team_deltas[side] / 5.0 + self.k_perf * basic_relative
                rich_deltas = team_deltas[side] / 5.0 + self.k_perf * rich_relative
                for position, player_key in enumerate(roster["player_key"].astype(str)):
                    self.player_ratings_basic[player_key] = float(
                        player_values["basic"][side][position] + basic_deltas[position]
                    )
                    self.player_ratings_rich[player_key] = float(
                        player_values["rich"][side][position] + rich_deltas[position]
                    )
                    self.player_ratings_decay[player_key] = float(
                        player_values["decay"][side][position] + rich_deltas[position]
                    )
                    self.player_last_seen[player_key] = now
                    self.player_map_counts[player_key] += 1
                    if bool(roster.iloc[position]["has_rich_stats"]):
                        self.player_rich_update_counts[player_key] += 1
                self.lineup_map_counts[lineup_keys[side]] += 1
            self.replayed_timestamps_.append(now)

        self.feature_frame_ = pd.DataFrame(feature_rows)
        self.as_of_date_ = cutoff
        last_timestamp = max(self.replayed_timestamps_)
        if last_timestamp >= cutoff:
            raise AssertionError("A replayed event reached or exceeded the cutoff.")
        return ReplayResult(
            as_of_date=cutoff,
            features=self.feature_frame_.copy(),
            replayed_rows=len(history),
            last_replayed_timestamp=last_timestamp,
        )

    def resolve_team(self, normalized_name: str) -> str:
        candidates = self.team_aliases.get(normalized_name, set())
        if not candidates:
            return normalized_name
        return max(
            candidates,
            key=lambda key: (
                self.team_map_counts[key],
                self.team_last_seen.get(key, pd.Timestamp.min),
                key,
            ),
        )

    def resolve_player(self, normalized_name: str) -> str:
        candidates = self.player_aliases.get(normalized_name, set())
        if not candidates:
            return f"name:{normalized_name}"
        return max(
            candidates,
            key=lambda key: (
                self.player_map_counts[key],
                self.player_last_seen.get(key, pd.Timestamp.min),
                key,
            ),
        )

    def team_status(self, team_key: str) -> dict[str, object]:
        return {
            "canonical_key": team_key,
            "n_games": int(self.team_map_counts[team_key]),
            "last_seen": self.team_last_seen.get(team_key),
            "cold_start": self.team_map_counts[team_key] == 0,
        }

    def player_status(self, player_key: str) -> dict[str, object]:
        return {
            "canonical_key": player_key,
            "n_games": int(self.player_map_counts[player_key]),
            "last_seen": self.player_last_seen.get(player_key),
            "cold_start": self.player_map_counts[player_key] == 0,
        }

    def build_matchup_feature_dict(
        self,
        team1_key: str,
        team2_key: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
        *,
        at: pd.Timestamp,
        player_scale_factor: float,
    ) -> dict[str, float]:
        """Build the locked V2 feature mapping without mutating replay state."""
        return _matchup_feature_dict(
            team1_key,
            team2_key,
            team1_players,
            team2_players,
            at=at,
            player_scale_factor=player_scale_factor,
            team_ratings=self.team_ratings,
            team_last_seen=self.team_last_seen,
            player_ratings=self.player_ratings_decay,
            player_last_seen=self.player_last_seen,
            player_map_counts=self.player_map_counts,
            lineup_map_counts=self.lineup_map_counts,
            team_half_life_days=self.team_half_life_days,
            player_half_life_days=self.player_half_life_days,
            initial_rating=self.initial_rating,
        )
