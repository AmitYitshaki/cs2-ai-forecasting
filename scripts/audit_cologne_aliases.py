"""Audit canonical team identity continuity for Cologne playoff teams."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELO_MODULE_DIRECTORY = PROJECT_ROOT / "src" / "features"
sys.path.insert(0, str(ELO_MODULE_DIRECTORY))

from elo import PointInTimeEngine  # noqa: E402


COLOGNE_KEYS = (
    "aurora",
    "betboom team",
    "9z team",
    "furia",
    "g2",
    "spirit",
    "falcons",
    "vitality",
)


def main() -> None:
    raw = pd.read_csv(
        PROJECT_ROOT / "data" / "final_tournament_features.csv",
        low_memory=False,
    )
    featured = PointInTimeEngine().transform(raw)
    test = featured.loc[featured["datetime"] >= pd.Timestamp("2026-04-01")]

    rows: list[dict[str, float | int | str]] = []
    for team_key in COLOGNE_KEYS:
        side1 = test.loc[
            test["team1_join_key"].eq(team_key),
            ["datetime", "team1_elo_global"],
        ].rename(columns={"team1_elo_global": "elo"})
        side2 = test.loc[
            test["team2_join_key"].eq(team_key),
            ["datetime", "team2_elo_global"],
        ].rename(columns={"team2_elo_global": "elo"})
        appearances = pd.concat([side1, side2]).sort_values("datetime")
        if appearances.empty:
            raise AssertionError(f"No Test rows found for {team_key}.")
        rows.append(
            {
                "team": team_key,
                "test_rows": len(appearances),
                "global_1500_rows": int(appearances["elo"].eq(1500.0).sum()),
                "first_test_elo": float(appearances.iloc[0]["elo"]),
            }
        )

    audit = pd.DataFrame(rows)
    if audit["global_1500_rows"].ne(0).any():
        raise AssertionError("A Cologne playoff team reset to 1500 in Test.")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
