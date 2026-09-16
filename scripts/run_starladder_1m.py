"""Reproducible one-million-path StarLadder production simulation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
import sys

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELO_MODULE_DIR = PROJECT_ROOT / "src" / "features"
if str(ELO_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(ELO_MODULE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elo import PointInTimeEngine
from src.models.bracket_simulator import (
    N_ELO_FEATURES_EXPECTED_ORDER,
    TEAM_DISPLAY_NAMES,
    TournamentAnalytics,
    TournamentEloState,
    print_tournament_summary,
    simulate_double_elimination_tournament,
)
from src.models.simulator import FrozenEloSnapshot
from src.models.tournament_exporter import export_tournament_results


EVENT_START = pd.Timestamp("2026-09-17 00:00:00")
K_FACTOR = 24.0
N_ITERATIONS = 1_000_000
PREFLIGHT_ITERATIONS = 5_000
BATCH_SIZE = 5_000
SEEDS = (42, 99)
GRAND_FINAL_RESET = False
OPENING_MATCHUPS = (
    ("mouz", "nrg"),
    ("vitality", "magic"),
    ("natus vincere", "aurora"),
    ("furia", "mibr"),
)
ACTIVE_MAP_POOL = (
    "Cache", "Dust2", "Mirage", "Inferno", "Nuke", "Ancient", "Anubis"
)
TOURNAMENT_TEAMS = tuple(team for pair in OPENING_MATCHUPS for team in pair)
MODEL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "map_classifier"
    / "canonical_elo_isotonic.joblib"
)


class SlowPandasPredictor:
    """Reference inference path used only for the pre-flight comparison."""

    def __init__(self, calibrated_model: object) -> None:
        self.calibrated_model = calibrated_model

    def predict_numpy(self, matrix) -> object:
        frame = pd.DataFrame(matrix, columns=N_ELO_FEATURES_EXPECTED_ORDER)
        return self.calibrated_model.predict_proba(frame)[:, 1]


def build_starting_state() -> tuple[TournamentEloState, pd.Timestamp]:
    raw = pd.read_csv(
        PROJECT_ROOT / "data" / "final_tournament_features.csv", low_memory=False
    )
    datetimes = pd.to_datetime(raw["datetime"], errors="raise")
    history = raw.loc[datetimes < EVENT_START].copy()
    engine = PointInTimeEngine(k_factor=K_FACTOR)
    engine.transform(history)
    snapshot = FrozenEloSnapshot(
        global_ratings=dict(engine.global_ratings_),
        map_ratings=dict(engine.map_ratings_),
        initial_rating=engine.initial_rating,
    )
    state = TournamentEloState.from_snapshot(
        snapshot, TOURNAMENT_TEAMS, ACTIVE_MAP_POOL, k_factor=K_FACTOR
    )
    cold_starts = [
        team
        for team in TOURNAMENT_TEAMS
        if state.global_rating(team) == snapshot.initial_rating
    ]
    assert not cold_starts, f"Unexpected 1500 Elo cold starts: {cold_starts}"
    return state, datetimes.loc[history.index].max()


def run_simulation(
    model: object,
    state: TournamentEloState,
    *,
    n_iterations: int,
    seed: int,
    show_progress: bool,
) -> tuple[dict, TournamentAnalytics, float]:
    started = perf_counter()
    counts, analytics = simulate_double_elimination_tournament(
        OPENING_MATCHUPS,
        state,
        model,
        ACTIVE_MAP_POOL,
        n_iterations=n_iterations,
        random_state=seed,
        batch_size=min(BATCH_SIZE, n_iterations),
        grand_final_reset=GRAND_FINAL_RESET,
        track_analytics=True,
        show_progress=show_progress,
    )
    return counts, analytics, perf_counter() - started


def assert_preflight_equivalence(model: object, state: TournamentEloState) -> None:
    fast_counts, fast_analytics, fast_seconds = run_simulation(
        model,
        state,
        n_iterations=PREFLIGHT_ITERATIONS,
        seed=314_159,
        show_progress=False,
    )
    slow_counts, slow_analytics, slow_seconds = run_simulation(
        SlowPandasPredictor(model),
        state,
        n_iterations=PREFLIGHT_ITERATIONS,
        seed=314_159,
        show_progress=False,
    )
    assert fast_counts == slow_counts
    assert fast_analytics.grand_final_matchups == slow_analytics.grand_final_matchups
    assert fast_analytics.runner_ups == slow_analytics.runner_ups
    assert fast_analytics.exact_podiums == slow_analytics.exact_podiums
    assert (
        fast_analytics.grand_final_appearances
        == slow_analytics.grand_final_appearances
    )
    print(
        "Pre-flight passed exactly: "
        f"fast={fast_seconds:.3f}s, slow={slow_seconds:.3f}s"
    )


def probability_map(counter: Counter, n_iterations: int) -> dict:
    return {key: count / n_iterations for key, count in counter.items()}


def max_probability_difference(
    first: Counter, second: Counter, n_iterations: int
) -> float:
    keys = set(first) | set(second)
    return max(
        (abs(first[key] - second[key]) / n_iterations for key in keys),
        default=0.0,
    )


def display_name(team: object) -> str:
    return TEAM_DISPLAY_NAMES.get(team, str(team))


def top_analytics_tables(
    analytics: TournamentAnalytics,
    counts: dict,
    n_iterations: int,
    limit: int = 3,
) -> dict[str, pd.DataFrame]:
    cinderella = analytics.cinderella_runs(
        {team: values["Champion"] for team, values in counts.items()},
        n_iterations,
    )
    return {
        "Grand Final Matchups": pd.DataFrame([
            {
                "Matchup": f"{display_name(pair[0])} vs {display_name(pair[1])}",
                "Probability": count / n_iterations,
            }
            for pair, count in analytics.grand_final_matchups.most_common(limit)
        ]),
        "Runner-Up": pd.DataFrame([
            {"Team": display_name(team), "Probability": count / n_iterations}
            for team, count in analytics.runner_ups.most_common(limit)
        ]),
        "Exact Podium [1st, 2nd, 3rd]": pd.DataFrame([
            {
                "Podium": " > ".join(display_name(team) for team in podium),
                "Probability": count / n_iterations,
            }
            for podium, count in analytics.exact_podiums.most_common(limit)
        ]),
        "Cinderella Grand Final Runs": pd.DataFrame([
            {"Team": display_name(team), "Probability": count / n_iterations}
            for team, count in cinderella.most_common(limit)
        ]),
    }


def analytics_metadata(
    analytics: TournamentAnalytics, counts: dict, n_iterations: int
) -> dict:
    champions = {team: values["Champion"] for team, values in counts.items()}
    cinderella = analytics.cinderella_runs(champions, n_iterations)

    def serialize(counter: Counter, labeler) -> list[dict]:
        return [
            {
                "key": labeler(key),
                "count": int(count),
                "probability": count / n_iterations,
            }
            for key, count in counter.most_common()
        ]

    return {
        "grand_final_matchups": serialize(
            analytics.grand_final_matchups,
            lambda pair: f"{pair[0]}__vs__{pair[1]}",
        ),
        "runner_ups": serialize(analytics.runner_ups, str),
        "exact_podiums": serialize(
            analytics.exact_podiums,
            lambda podium: "__then__".join(map(str, podium)),
        ),
        "cinderella_definition": "P(Champion) < 0.10 in this run",
        "cinderella_grand_final_runs": serialize(cinderella, str),
    }


def main() -> dict:
    state, last_history_timestamp = build_starting_state()
    bundle = joblib.load(MODEL_ARTIFACT_PATH)
    assert bundle["feature_columns"] == list(N_ELO_FEATURES_EXPECTED_ORDER)
    deployed_model = bundle["calibrated_model"]
    print(f"Last historical map: {last_history_timestamp}")
    print({team: round(state.global_rating(team), 3) for team in TOURNAMENT_TEAMS})
    assert_preflight_equivalence(deployed_model, state)

    runs: dict[int, dict] = {}
    for seed in SEEDS:
        counts, analytics, seconds = run_simulation(
            deployed_model,
            state,
            n_iterations=N_ITERATIONS,
            seed=seed,
            show_progress=True,
        )
        print(f"seed={seed} execution_seconds={seconds:.3f}")
        summary = print_tournament_summary(counts, N_ITERATIONS)
        tables = top_analytics_tables(analytics, counts, N_ITERATIONS)
        for category, table in tables.items():
            print(f"\nTop {len(table)} — {category}")
            printable = table.copy()
            printable["Probability"] = printable["Probability"].map(
                lambda probability: f"{probability:.4%}"
            )
            print(printable.to_string(index=False))
        runs[seed] = {
            "counts": counts,
            "analytics": analytics,
            "seconds": seconds,
            "summary": summary,
            "tables": tables,
        }

    counts_42 = runs[42]["counts"]
    counts_99 = runs[99]["counts"]
    analytics_42 = runs[42]["analytics"]
    analytics_99 = runs[99]["analytics"]
    champion_42 = Counter({t: v["Champion"] for t, v in counts_42.items()})
    champion_99 = Counter({t: v["Champion"] for t, v in counts_99.items()})
    cinderella_42 = analytics_42.cinderella_runs(champion_42, N_ITERATIONS)
    cinderella_99 = analytics_99.cinderella_runs(champion_99, N_ITERATIONS)
    stability = {
        "P(Champion)": max_probability_difference(
            champion_42, champion_99, N_ITERATIONS
        ),
        "Grand Final Matchup": max_probability_difference(
            analytics_42.grand_final_matchups,
            analytics_99.grand_final_matchups,
            N_ITERATIONS,
        ),
        "Runner-Up": max_probability_difference(
            analytics_42.runner_ups, analytics_99.runner_ups, N_ITERATIONS
        ),
        "Exact Podium": max_probability_difference(
            analytics_42.exact_podiums, analytics_99.exact_podiums, N_ITERATIONS
        ),
        "Cinderella Grand Final": max_probability_difference(
            cinderella_42, cinderella_99, N_ITERATIONS
        ),
    }
    print("\nSeed stability")
    for category, difference in stability.items():
        print(f"{category}: max absolute difference={difference:.4%}")
    assert max(stability.values()) <= 0.005, stability

    for seed in SEEDS:
        exported = export_tournament_results(
            runs[seed]["summary"],
            "starladder",
            PROJECT_ROOT / "results",
            seed,
            N_ITERATIONS,
            MODEL_ARTIFACT_PATH,
            k_factor=K_FACTOR,
            active_map_pool=ACTIVE_MAP_POOL,
            quarterfinals=OPENING_MATCHUPS,
            historical_map_orders={},
            execution_seconds=runs[seed]["seconds"],
            extra_metadata={
                "format": "eight_team_double_elimination",
                "grand_final_best_of": 5,
                "other_matches_best_of": 3,
                "grand_final_reset": GRAND_FINAL_RESET,
                "elo_snapshot_cutoff": EVENT_START.isoformat(),
                "last_historical_map": last_history_timestamp.isoformat(),
                "preflight_iterations": PREFLIGHT_ITERATIONS,
                "seed_stability_max_absolute_differences": stability,
                "advanced_analytics": analytics_metadata(
                    runs[seed]["analytics"], runs[seed]["counts"], N_ITERATIONS
                ),
            },
        )
        runs[seed]["exported"] = exported
        print(f"seed={seed} metadata={exported.metadata_path.relative_to(PROJECT_ROOT)}")
    return runs


if __name__ == "__main__":
    main()
