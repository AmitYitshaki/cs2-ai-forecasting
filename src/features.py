"""Leakage-safe preparation and feature engineering for CS2 map rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterAudit:
    """Row counts produced while constructing the map-level allowlist."""

    raw_rows: int
    missing_map_rows: int
    phantom_rows: int
    non_positive_score_rows: int
    valid_rows: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation of the audit."""

        return asdict(self)


class MapDatasetPreparer:
    """Filter invalid rows and retain only approved pre-modeling columns."""

    IDENTIFIER_COLUMNS: ClassVar[tuple[str, ...]] = (
        "match_id",
        "game_id",
        "tournament",
        "team1_id",
        "team1",
        "team2_id",
        "team2",
        "map_id",
        "team1_join_key",
        "team2_join_key",
    )
    CONTEXT_COLUMNS: ClassVar[tuple[str, ...]] = (
        "datetime",
        "map_name",
        "team1_win",
    )
    TEAM1_DNA_COLUMNS: ClassVar[tuple[str, ...]] = (
        "team1_pistol_round_win_rate",
        "team1_n_pistol_rounds",
        "team1_ct_win_rate",
        "team1_n_ct_rounds",
        "team1_t_win_rate",
        "team1_n_t_rounds",
    )
    TEAM2_DNA_COLUMNS: ClassVar[tuple[str, ...]] = tuple(
        column.replace("team1_", "team2_") for column in TEAM1_DNA_COLUMNS
    )
    FILTER_COLUMNS: ClassVar[tuple[str, ...]] = (
        "is_total",
        "bestOf",
        "score1_game",
        "score2_game",
    )
    POST_MATCH_TOKENS: ClassVar[tuple[str, ...]] = (
        "score",
        "kills",
        "deaths",
        "assists",
        "adr",
        "kast",
        "kddiff",
        "games_played",
        "player1",
        "player2",
        "player3",
        "player4",
        "player5",
    )

    @property
    def source_allowlist(self) -> tuple[str, ...]:
        """Columns allowed to survive filtering before feature engineering."""

        return (
            self.IDENTIFIER_COLUMNS
            + self.CONTEXT_COLUMNS
            + self.TEAM1_DNA_COLUMNS
            + self.TEAM2_DNA_COLUMNS
        )

    def transform(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, FilterAudit]:
        """Return valid map rows and an audit of every removal reason.

        The score fields are used only as a validity check and are discarded by
        the explicit allowlist before the dataframe is returned.
        """

        self._validate_required_columns(raw_df)

        is_total = self._coerce_boolean(raw_df["is_total"], "is_total")
        best_of_one = pd.to_numeric(raw_df["bestOf"], errors="coerce").eq(1)
        map_present = raw_df["map_name"].notna()

        score1 = pd.to_numeric(raw_df["score1_game"], errors="coerce")
        score2 = pd.to_numeric(raw_df["score2_game"], errors="coerce")
        positive_score = (score1 + score2).gt(0)

        phantom_rows = map_present & is_total & ~best_of_one
        filter_candidate = map_present & (~is_total | best_of_one)
        non_positive_score_rows = filter_candidate & ~positive_score
        valid_mask = filter_candidate & positive_score

        validation_df = raw_df.loc[
            valid_mask,
            ["map_name", "is_total", "bestOf", "score1_game", "score2_game"],
        ].copy()
        validation_is_total = self._coerce_boolean(
            validation_df["is_total"], "is_total"
        )
        validation_best_of_one = pd.to_numeric(
            validation_df["bestOf"], errors="coerce"
        ).eq(1)
        validation_score = pd.to_numeric(
            validation_df["score1_game"], errors="coerce"
        ) + pd.to_numeric(validation_df["score2_game"], errors="coerce")

        if validation_df["map_name"].isna().any():
            raise AssertionError("A retained row has no map_name.")
        if (validation_is_total & ~validation_best_of_one).any():
            raise AssertionError("A phantom is_total Bo3/Bo5 row survived.")
        if not validation_score.gt(0).all():
            raise AssertionError("A retained row has a non-positive map score.")

        prepared_df = raw_df.loc[valid_mask, self.source_allowlist].copy()
        prepared_df["datetime"] = pd.to_datetime(
            prepared_df["datetime"], errors="raise"
        )
        prepared_df["team1_has_dna"] = prepared_df[
            list(self.TEAM1_DNA_COLUMNS)
        ].notna().all(axis=1)
        prepared_df["team2_has_dna"] = prepared_df[
            list(self.TEAM2_DNA_COLUMNS)
        ].notna().all(axis=1)

        self._validate_prepared_dataframe(prepared_df)
        prepared_df = prepared_df.sort_values(
            ["datetime", "match_id", "game_id"], kind="stable"
        ).reset_index(drop=True)

        audit = FilterAudit(
            raw_rows=len(raw_df),
            missing_map_rows=int((~map_present).sum()),
            phantom_rows=int(phantom_rows.sum()),
            non_positive_score_rows=int(non_positive_score_rows.sum()),
            valid_rows=len(prepared_df),
        )
        return prepared_df, audit

    def _validate_required_columns(self, raw_df: pd.DataFrame) -> None:
        required_columns = set(self.source_allowlist + self.FILTER_COLUMNS)
        missing_columns = sorted(required_columns - set(raw_df.columns))
        if missing_columns:
            raise ValueError(f"Required columns are missing: {missing_columns}")

    def _validate_prepared_dataframe(self, prepared_df: pd.DataFrame) -> None:
        leaked_columns = [
            column
            for column in prepared_df.columns
            if any(token in column.lower() for token in self.POST_MATCH_TOKENS)
        ]
        if leaked_columns:
            raise AssertionError(
                f"Post-match columns passed the allowlist: {leaked_columns}"
            )
        if prepared_df["map_name"].isna().any():
            raise AssertionError("Prepared dataframe contains missing map names.")
        if prepared_df["datetime"].isna().any():
            raise AssertionError("Prepared dataframe contains invalid datetimes.")
        if not prepared_df["team1_win"].isin([0, 1]).all():
            raise ValueError("team1_win contains values outside {0, 1}.")

    @staticmethod
    def _coerce_boolean(series: pd.Series, column_name: str) -> pd.Series:
        if pd.api.types.is_bool_dtype(series.dtype):
            return series.fillna(False).astype(bool)

        normalized = series.astype("string").str.strip().str.lower()
        mapping = {
            "true": True,
            "1": True,
            "false": False,
            "0": False,
        }
        invalid_values = sorted(
            normalized[normalized.notna() & ~normalized.isin(mapping)].unique()
        )
        if invalid_values:
            raise ValueError(
                f"{column_name} contains invalid boolean values: {invalid_values}"
            )
        return normalized.map(mapping).fillna(False).astype(bool)


class SymmetricFeatureEngineer:
    """Symmetrize one chronological split and add relative DNA features."""

    SWAP_COLUMN_PAIRS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("team1_id", "team2_id"),
        ("team1", "team2"),
        ("team1_join_key", "team2_join_key"),
        ("team1_pistol_round_win_rate", "team2_pistol_round_win_rate"),
        ("team1_n_pistol_rounds", "team2_n_pistol_rounds"),
        ("team1_ct_win_rate", "team2_ct_win_rate"),
        ("team1_n_ct_rounds", "team2_n_ct_rounds"),
        ("team1_t_win_rate", "team2_t_win_rate"),
        ("team1_n_t_rounds", "team2_n_t_rounds"),
        ("team1_has_dna", "team2_has_dna"),
    )
    DIFF_FEATURE_PAIRS: ClassVar[dict[str, tuple[str, str]]] = {
        "diff_pistol_win_rate": (
            "team1_pistol_round_win_rate",
            "team2_pistol_round_win_rate",
        ),
        "diff_n_pistol_rounds": (
            "team1_n_pistol_rounds",
            "team2_n_pistol_rounds",
        ),
        "diff_ct_win_rate": ("team1_ct_win_rate", "team2_ct_win_rate"),
        "diff_n_ct_rounds": ("team1_n_ct_rounds", "team2_n_ct_rounds"),
        "diff_t_win_rate": ("team1_t_win_rate", "team2_t_win_rate"),
        "diff_n_t_rounds": ("team1_n_t_rounds", "team2_n_t_rounds"),
    }

    def transform(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Symmetrize and then add differences to one isolated split."""

        return self.add_diff_features(self.symmetrize(split_df))

    def symmetrize(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Append a mirrored copy with teams swapped and target flipped."""

        self._validate_input(split_df)
        mirrored_df = split_df.copy()

        for team1_column, team2_column in self.SWAP_COLUMN_PAIRS:
            mirrored_df[team1_column] = split_df[team2_column].to_numpy(copy=True)
            mirrored_df[team2_column] = split_df[team1_column].to_numpy(copy=True)
        mirrored_df["team1_win"] = 1 - split_df["team1_win"].to_numpy()

        self._validate_mirror(split_df, mirrored_df)
        symmetrized_df = pd.concat([split_df, mirrored_df], ignore_index=True)
        if len(symmetrized_df) != 2 * len(split_df):
            raise AssertionError("Symmetrization did not exactly double row count.")
        if not np.isclose(symmetrized_df["team1_win"].mean(), 0.5):
            raise AssertionError("Symmetrization did not balance the target.")
        return symmetrized_df

    def add_diff_features(self, symmetrized_df: pd.DataFrame) -> pd.DataFrame:
        """Add explicit team1-minus-team2 DNA features."""

        engineered_df = symmetrized_df.copy()
        for diff_column, (
            team1_column,
            team2_column,
        ) in self.DIFF_FEATURE_PAIRS.items():
            engineered_df[diff_column] = (
                engineered_df[team1_column] - engineered_df[team2_column]
            )

        midpoint = len(engineered_df) // 2
        for diff_column in self.DIFF_FEATURE_PAIRS:
            original_values = engineered_df.iloc[:midpoint][diff_column].to_numpy()
            mirrored_values = engineered_df.iloc[midpoint:][diff_column].to_numpy()
            if not np.allclose(
                original_values, -mirrored_values, equal_nan=True
            ):
                raise AssertionError(
                    f"Diff feature is not antisymmetric: {diff_column}"
                )
        return engineered_df

    def _validate_input(self, split_df: pd.DataFrame) -> None:
        required_columns = {
            "team1_win",
            *(column for pair in self.SWAP_COLUMN_PAIRS for column in pair),
        }
        missing_columns = sorted(required_columns - set(split_df.columns))
        if missing_columns:
            raise ValueError(
                f"Columns required for symmetrization are missing: {missing_columns}"
            )
        if not split_df["team1_win"].isin([0, 1]).all():
            raise ValueError("team1_win must be binary before symmetrization.")

    def _validate_mirror(
        self, original_df: pd.DataFrame, mirrored_df: pd.DataFrame
    ) -> None:
        for team1_column, team2_column in self.SWAP_COLUMN_PAIRS:
            pd.testing.assert_series_equal(
                mirrored_df[team1_column],
                original_df[team2_column],
                check_names=False,
            )
            pd.testing.assert_series_equal(
                mirrored_df[team2_column],
                original_df[team1_column],
                check_names=False,
            )
        expected_target = 1 - original_df["team1_win"]
        pd.testing.assert_series_equal(
            mirrored_df["team1_win"], expected_target, check_names=False
        )
