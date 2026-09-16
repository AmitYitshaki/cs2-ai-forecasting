"""Strict chronological splitting for the CS2 map-classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplitConfig:
    """Exclusive timestamp boundaries for Train, Validation, and Test."""

    validation_start: pd.Timestamp = pd.Timestamp("2026-01-01")
    test_start: pd.Timestamp = pd.Timestamp("2026-04-01")


class ChronologicalSplitter:
    """Split valid map rows before any row-multiplying feature engineering."""

    def __init__(self, config: ChronologicalSplitConfig | None = None) -> None:
        self.config = config or ChronologicalSplitConfig()

    def split(self, dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Return mutually exclusive chronological Train, Val, and Test frames."""

        if "datetime" not in dataframe.columns:
            raise ValueError("datetime is required for chronological splitting.")

        datetimes = pd.to_datetime(dataframe["datetime"], errors="raise")
        if datetimes.isna().any():
            raise ValueError("datetime contains missing values.")

        validation_start = self.config.validation_start
        test_start = self.config.test_start
        if validation_start >= test_start:
            raise ValueError("validation_start must precede test_start.")

        masks = {
            "train": datetimes < validation_start,
            "val": (datetimes >= validation_start) & (datetimes < test_start),
            "test": datetimes >= test_start,
        }
        assignment_count = sum(mask.astype("int8") for mask in masks.values())
        if not assignment_count.eq(1).all():
            raise AssertionError("Every row must belong to exactly one split.")

        splits = {
            name: dataframe.loc[mask].copy().reset_index(drop=True)
            for name, mask in masks.items()
        }
        self._validate_boundaries(splits)
        self._validate_identifier_separation(splits, "game_id")
        return splits

    def _validate_boundaries(self, splits: dict[str, pd.DataFrame]) -> None:
        validation_start = self.config.validation_start
        test_start = self.config.test_start

        if not (splits["train"]["datetime"] < validation_start).all():
            raise AssertionError("Train contains a row from 2026 or later.")
        if not (
            (splits["val"]["datetime"] >= validation_start)
            & (splits["val"]["datetime"] < test_start)
        ).all():
            raise AssertionError("Validation contains a row outside 2026 Q1.")
        if not (splits["test"]["datetime"] >= test_start).all():
            raise AssertionError("Test contains a row before 2026 Q2.")

    @staticmethod
    def _validate_identifier_separation(
        splits: dict[str, pd.DataFrame], identifier: str
    ) -> None:
        if identifier not in next(iter(splits.values())).columns:
            return
        identifiers = {
            name: set(split_df[identifier].dropna().tolist())
            for name, split_df in splits.items()
        }
        pairs = (("train", "val"), ("train", "test"), ("val", "test"))
        for left_name, right_name in pairs:
            overlap = identifiers[left_name] & identifiers[right_name]
            if overlap:
                raise AssertionError(
                    f"{identifier} overlaps {left_name}/{right_name}: "
                    f"{sorted(overlap)[:5]}"
                )
