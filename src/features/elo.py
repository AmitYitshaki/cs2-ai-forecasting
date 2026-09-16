"""Point-in-time Elo and head-to-head features for CS2 map results."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import ClassVar

import pandas as pd


class PointInTimeEngine:
    """Build leakage-safe Elo and H2H features in chronological order.

    Each output row contains ratings observed immediately before that map. The
    result of the current map is applied only after its features are recorded.
    H2H results are deferred until every map in the current match is recorded,
    so a row never sees an outcome from its own match in its H2H counters.
    Calling :meth:`transform` starts from fresh ratings, making repeated calls
    deterministic and preventing state from leaking across datasets.
    """

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "match_id",
        "game_id",
        "datetime",
        "map_name",
        "is_total",
        "bestOf",
        "score1_game",
        "score2_game",
        "team1_id",
        "team2_id",
        "team1_win",
    )
    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "team1_elo_global",
        "team2_elo_global",
        "team1_elo_map",
        "team2_elo_map",
        "team1_map_elo_n_games",
        "team2_map_elo_n_games",
        "raw_global_elo_prob",
        "raw_map_elo_prob",
        "team1_h2h_wins",
        "team2_h2h_wins",
        "team1_roll5_win_pct",
        "team2_roll5_win_pct",
        "team1_roll5_round_diff",
        "team2_roll5_round_diff",
    )

    def __init__(self, k_factor: float = 24.0, initial_rating: float = 1500.0) -> None:
        if k_factor <= 0:
            raise ValueError("k_factor must be positive.")
        if initial_rating <= 0:
            raise ValueError("initial_rating must be positive.")

        self.k_factor = float(k_factor)
        self.initial_rating = float(initial_rating)
        self.reset()

    def reset(self) -> None:
        """Reset all learned ratings, map-game counts, and H2H results."""

        self.global_ratings_: dict[Hashable, float] = {}
        self.map_ratings_: dict[tuple[Hashable, Hashable], float] = {}
        self.map_game_counts_: dict[tuple[Hashable, Hashable], int] = {}
        self.h2h_wins_: dict[tuple[Hashable, Hashable], int] = {}
        self.rolling_history_: dict[
            Hashable, deque[tuple[float, float]]
        ] = {}

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Filter valid maps and append pre-map Elo features.

        The input is copied, parsed, filtered with the locked phantom-row rule,
        and sorted by ``datetime`` with ``match_id`` and ``game_id`` as stable
        tie-breakers. Ratings are always reset before the chronological pass.
        """

        self._validate_columns(dataframe)
        prepared = self._prepare(dataframe)
        self.reset()
        h2h_rows = self._build_h2h_features(prepared)
        rolling_rows = self._build_rolling_features(prepared)

        feature_rows: list[dict[str, float | int]] = []
        for row, h2h_features, rolling_features in zip(
            prepared.itertuples(index=False),
            h2h_rows,
            rolling_rows,
            strict=True,
        ):
            row_values = row._asdict()
            team1_id = self._team_identity(row_values, 1)
            team2_id = self._team_identity(row_values, 2)
            map_name = row.map_name
            actual = float(row.team1_win)

            team1_global = self.global_ratings_.get(team1_id, self.initial_rating)
            team2_global = self.global_ratings_.get(team2_id, self.initial_rating)
            team1_map_key = (team1_id, map_name)
            team2_map_key = (team2_id, map_name)
            team1_map = self.map_ratings_.get(team1_map_key, self.initial_rating)
            team2_map = self.map_ratings_.get(team2_map_key, self.initial_rating)
            team1_map_games = self.map_game_counts_.get(team1_map_key, 0)
            team2_map_games = self.map_game_counts_.get(team2_map_key, 0)

            global_probability = self.expected_score(team1_global, team2_global)
            map_probability = self.expected_score(team1_map, team2_map)
            feature_rows.append(
                {
                    "team1_elo_global": team1_global,
                    "team2_elo_global": team2_global,
                    "team1_elo_map": team1_map,
                    "team2_elo_map": team2_map,
                    "team1_map_elo_n_games": team1_map_games,
                    "team2_map_elo_n_games": team2_map_games,
                    "raw_global_elo_prob": global_probability,
                    "raw_map_elo_prob": map_probability,
                    **h2h_features,
                    **rolling_features,
                }
            )

            global_delta = self.k_factor * (actual - global_probability)
            self.global_ratings_[team1_id] = team1_global + global_delta
            self.global_ratings_[team2_id] = team2_global - global_delta

            map_delta = self.k_factor * (actual - map_probability)
            self.map_ratings_[team1_map_key] = team1_map + map_delta
            self.map_ratings_[team2_map_key] = team2_map - map_delta
            self.map_game_counts_[team1_map_key] = team1_map_games + 1
            self.map_game_counts_[team2_map_key] = team2_map_games + 1

        features = pd.DataFrame(feature_rows, columns=self.FEATURE_COLUMNS)
        result = pd.concat([prepared.reset_index(drop=True), features], axis=1)
        self._validate_output(result)
        return result

    def _build_h2h_features(
        self, prepared: pd.DataFrame
    ) -> list[dict[str, int]]:
        """Return H2H wins known strictly before each row's match."""

        h2h_rows: list[dict[str, int] | None] = [None] * len(prepared)
        for _, match_rows in prepared.groupby("match_id", sort=False):
            for row_index, row in match_rows.iterrows():
                team1_id = self._team_identity(row, 1)
                team2_id = self._team_identity(row, 2)
                h2h_rows[row_index] = {
                    "team1_h2h_wins": self.h2h_wins_.get(
                        (team1_id, team2_id), 0
                    ),
                    "team2_h2h_wins": self.h2h_wins_.get(
                        (team2_id, team1_id), 0
                    ),
                }

            # Commit all map results only after the current match features exist.
            for _, row in match_rows.iterrows():
                actual = int(row["team1_win"])
                team1_id = self._team_identity(row, 1)
                team2_id = self._team_identity(row, 2)
                forward_key = (team1_id, team2_id)
                reverse_key = (team2_id, team1_id)
                self.h2h_wins_[forward_key] = (
                    self.h2h_wins_.get(forward_key, 0) + actual
                )
                self.h2h_wins_[reverse_key] = (
                    self.h2h_wins_.get(reverse_key, 0) + (1 - actual)
                )

        if any(row is None for row in h2h_rows):
            raise AssertionError("H2H features were not assigned to every row.")
        return [row for row in h2h_rows if row is not None]

    def _build_rolling_features(
        self, prepared: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Return last-five-map form known before each row's match."""

        rolling_rows: list[dict[str, float] | None] = [None] * len(prepared)
        for _, match_rows in prepared.groupby("match_id", sort=False):
            for row_index, row in match_rows.iterrows():
                team1_id = self._team_identity(row, 1)
                team2_id = self._team_identity(row, 2)
                team1_history = self.rolling_history_.get(team1_id, deque())
                team2_history = self.rolling_history_.get(team2_id, deque())
                rolling_rows[row_index] = {
                    "team1_roll5_win_pct": self._rolling_win_rate(team1_history),
                    "team2_roll5_win_pct": self._rolling_win_rate(team2_history),
                    "team1_roll5_round_diff": self._rolling_round_diff(
                        team1_history
                    ),
                    "team2_roll5_round_diff": self._rolling_round_diff(
                        team2_history
                    ),
                }

            # Defer updates so every map in the match sees only prior matches.
            for _, row in match_rows.iterrows():
                team1_id = self._team_identity(row, 1)
                team2_id = self._team_identity(row, 2)
                score1 = float(row["score1_game"])
                score2 = float(row["score2_game"])
                team1_result = float(score1 > score2)
                team2_result = 1.0 - team1_result
                round_diff = score1 - score2
                self.rolling_history_.setdefault(
                    team1_id, deque(maxlen=5)
                ).append((team1_result, round_diff))
                self.rolling_history_.setdefault(
                    team2_id, deque(maxlen=5)
                ).append((team2_result, -round_diff))

        if any(row is None for row in rolling_rows):
            raise AssertionError("Rolling features were not assigned to every row.")
        return [row for row in rolling_rows if row is not None]

    @staticmethod
    def _rolling_win_rate(history: deque[tuple[float, float]]) -> float:
        if not history:
            return 0.0
        return sum(result for result, _ in history) / len(history)

    @staticmethod
    def _rolling_round_diff(history: deque[tuple[float, float]]) -> float:
        return sum(round_diff for _, round_diff in history)

    @staticmethod
    def _team_identity(row: object, side: int) -> Hashable:
        """Prefer a canonical join key and fall back to the source team id."""

        canonical_column = f"team{side}_join_key"
        id_column = f"team{side}_id"
        if hasattr(row, "get"):
            canonical_value = row.get(canonical_column)
            id_value = row.get(id_column)
        else:
            canonical_value = None
            id_value = None

        if pd.notna(canonical_value) and str(canonical_value).strip():
            return str(canonical_value).strip().casefold()
        if pd.isna(id_value):
            raise ValueError(f"No stable identity is available for team side {side}.")
        return id_value

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Return the standard Elo win probability for player or team A."""

        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        prepared = dataframe.copy()
        prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="raise")
        if prepared["datetime"].isna().any():
            raise ValueError("datetime contains missing values.")

        is_total = self._coerce_boolean(prepared["is_total"])
        best_of_one = pd.to_numeric(prepared["bestOf"], errors="coerce").eq(1)
        score1 = pd.to_numeric(prepared["score1_game"], errors="coerce")
        score2 = pd.to_numeric(prepared["score2_game"], errors="coerce")
        positive_score = (score1 + score2).gt(0)
        phantom_filter = prepared["map_name"].notna() & (~is_total | best_of_one)
        valid_mask = phantom_filter & positive_score
        prepared = prepared.loc[valid_mask].copy()

        retained_score1 = pd.to_numeric(prepared["score1_game"], errors="raise")
        retained_score2 = pd.to_numeric(prepared["score2_game"], errors="raise")
        if retained_score1.eq(retained_score2).any():
            raise ValueError("A retained map has a tied score.")

        if prepared[["team1_id", "team2_id"]].isna().any().any():
            raise ValueError("Team identifiers contain missing values.")
        if not prepared["team1_win"].isin([0, 1]).all():
            raise ValueError("team1_win must contain only 0 and 1.")

        if prepared["game_id"].isna().any():
            raise ValueError("game_id contains missing values.")
        if prepared.duplicated(["match_id", "game_id"]).any():
            raise ValueError("game_id must be unique within each match_id.")

        prepared = prepared.sort_values(
            ["datetime", "match_id", "game_id"], kind="stable"
        ).reset_index(drop=True)
        if not prepared["datetime"].is_monotonic_increasing:
            raise AssertionError("Rows are not ordered chronologically.")
        if prepared.groupby("match_id")["datetime"].nunique().gt(1).any():
            raise ValueError("A match_id appears at multiple timestamps.")
        return prepared

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        missing = sorted(set(self.REQUIRED_COLUMNS) - set(dataframe.columns))
        if missing:
            raise ValueError(f"Required Elo columns are missing: {missing}")

    def _validate_output(self, result: pd.DataFrame) -> None:
        if result["map_name"].isna().any():
            raise AssertionError("A row without map_name survived Elo preparation.")
        is_total = self._coerce_boolean(result["is_total"])
        best_of_one = pd.to_numeric(result["bestOf"], errors="coerce").eq(1)
        if (is_total & ~best_of_one).any():
            raise AssertionError("A phantom is_total Bo3/Bo5 row survived Elo preparation.")
        score1 = pd.to_numeric(result["score1_game"], errors="coerce")
        score2 = pd.to_numeric(result["score2_game"], errors="coerce")
        if not (score1 + score2).gt(0).all():
            raise AssertionError("A row with a non-positive map score survived Elo preparation.")
        probabilities = result[["raw_global_elo_prob", "raw_map_elo_prob"]]
        if not probabilities.ge(0.0).all().all() or not probabilities.le(1.0).all().all():
            raise AssertionError("Elo probabilities must lie in [0, 1].")
        if (result[["team1_map_elo_n_games", "team2_map_elo_n_games"]] < 0).any().any():
            raise AssertionError("Map game counts cannot be negative.")
        if (result[["team1_h2h_wins", "team2_h2h_wins"]] < 0).any().any():
            raise AssertionError("H2H win counts cannot be negative.")
        rolling_win_rates = result[["team1_roll5_win_pct", "team2_roll5_win_pct"]]
        if not rolling_win_rates.ge(0.0).all().all():
            raise AssertionError("Rolling win rates cannot be negative.")
        if not rolling_win_rates.le(1.0).all().all():
            raise AssertionError("Rolling win rates cannot exceed one.")

    @staticmethod
    def _coerce_boolean(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series.dtype):
            return series.fillna(False).astype(bool)

        normalized = series.astype("string").str.strip().str.lower()
        mapping = {"true": True, "1": True, "false": False, "0": False}
        invalid = sorted(normalized[normalized.notna() & ~normalized.isin(mapping)].unique())
        if invalid:
            raise ValueError(f"is_total contains invalid boolean values: {invalid}")
        return normalized.map(mapping).fillna(False).astype(bool)


# Backward-compatible name used by the Elo-only ablation notebook.
EloEngine = PointInTimeEngine
