"""Derive StarLadder opening-series probabilities from the locked 1M run.

The production metadata contains aggregated advanced analytics, but it does not
contain per-iteration bracket paths.  This script therefore computes only the
opening Bo3 probabilities from the exact locked model and Elo snapshot.  It
records the unavailable full-path request explicitly instead of inventing or
re-simulating those data.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import sys
from typing import Hashable, Sequence

import joblib
import numpy as np


PROJECT_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

from scripts.run_starladder_1m import (
    ACTIVE_MAP_POOL,
    EVENT_START,
    K_FACTOR,
    MODEL_ARTIFACT_PATH,
    OPENING_MATCHUPS,
    PROJECT_ROOT,
    build_starting_state,
)
from src.models.bracket_simulator import (
    FastIsotonicXGBoostPredictor,
    N_ELO_FEATURES_EXPECTED_ORDER,
    TEAM_DISPLAY_NAMES,
    TournamentEloState,
)


SOURCE_STEM = "starladder_20260915T160338Z"
REPLAY_STEM = "starladder_20260916T112636Z"
SOURCE_SEEDS = (42, 99)
SOURCE_ITERATIONS = 1_000_000
REPORTED_OPENING_MATCHUPS = (
    ("mouz", "nrg"),
    ("natus vincere", "aurora"),
    ("vitality", "magic"),
    ("furia", "mibr"),
)
EXPECTED_MODEL_SHA256 = (
    "274a8118ada2c9551f7e668f8999fa0ff672b0158b6d055056a152f09820370d"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "starladder"
    / f"{SOURCE_STEM}_additional_outputs.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_validate_source_metadata(stem: str) -> list[tuple[Path, dict]]:
    sources: list[tuple[Path, dict]] = []
    for seed in SOURCE_SEEDS:
        path = (
            PROJECT_ROOT
            / "results"
            / "starladder"
            / f"{stem}_seed{seed}_n{SOURCE_ITERATIONS}_metadata.json"
        )
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["seed"] == seed
        assert metadata["n_iterations"] == SOURCE_ITERATIONS
        assert metadata["model_artifact_sha256"] == EXPECTED_MODEL_SHA256
        assert tuple(metadata["active_map_pool"]) == ACTIVE_MAP_POOL
        assert tuple(map(tuple, metadata["quarterfinals"])) == OPENING_MATCHUPS
        assert metadata["extra_metadata"]["elo_snapshot_cutoff"] == (
            EVENT_START.isoformat()
        )
        sources.append((path, metadata))
    return sources


def _updated_state(
    state: TournamentEloState,
    team_a: Hashable,
    team_b: Hashable,
    map_name: Hashable,
    ratings: np.ndarray,
    actual_a: float,
) -> TournamentEloState:
    result = deepcopy(state)
    rating_a, rating_b, map_rating_a, map_rating_b = ratings
    global_expected = 1.0 / (
        1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
    )
    global_delta = result.k_factor * (actual_a - global_expected)
    result.global_ratings[team_a] = rating_a + global_delta
    result.global_ratings[team_b] = rating_b - global_delta

    map_expected = 1.0 / (
        1.0 + 10.0 ** ((map_rating_b - map_rating_a) / 400.0)
    )
    map_delta = result.k_factor * (actual_a - map_expected)
    result.map_ratings[(team_a, map_name)] = map_rating_a + map_delta
    result.map_ratings[(team_b, map_name)] = map_rating_b - map_delta
    return result


def exact_unknown_map_bo3_probability(
    team_a: Hashable,
    team_b: Hashable,
    starting_state: TournamentEloState,
    predictor: FastIsotonicXGBoostPredictor,
) -> float:
    """Average exactly over every uniformly sampled ordered Bo3 map sequence."""

    ordered_maps = tuple(permutations(ACTIVE_MAP_POOL, 3))
    initial_weight = 1.0 / len(ordered_maps)
    scenarios = [
        (deepcopy(starting_state), map_order, 0, 0, initial_weight)
        for map_order in ordered_maps
    ]
    team_a_win_probability = 0.0

    # Breadth-first enumeration gives the exact probability while batching all
    # XGBoost calls for a map round into one fast NumPy inference operation.
    for map_index in range(3):
        matrix = np.empty((len(scenarios), 6), dtype=np.float32)
        map_names: list[Hashable] = []
        for row_index, (state, map_order, _, _, _) in enumerate(scenarios):
            map_name = map_order[map_index]
            rating_a = state.global_rating(team_a)
            rating_b = state.global_rating(team_b)
            map_rating_a = state.map_rating(team_a, map_name)
            map_rating_b = state.map_rating(team_b, map_name)
            matrix[row_index] = (
                rating_a,
                rating_b,
                map_rating_a,
                map_rating_b,
                rating_a - rating_b,
                map_rating_a - map_rating_b,
            )
            map_names.append(map_name)

        probabilities = predictor.predict_numpy(matrix)
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0.0) | (probabilities > 1.0)
        ).any():
            raise ValueError("The calibrated model returned invalid probabilities.")

        next_scenarios = []
        for row_index, (state, map_order, wins_a, wins_b, weight) in enumerate(
            scenarios
        ):
            probability_a = float(probabilities[row_index])
            map_name = map_names[row_index]
            ratings = matrix[row_index, :4]
            for actual_a, branch_probability in (
                (1.0, probability_a),
                (0.0, 1.0 - probability_a),
            ):
                branch_weight = weight * branch_probability
                branch_wins_a = wins_a + int(actual_a)
                branch_wins_b = wins_b + int(not actual_a)
                if branch_wins_a == 2:
                    team_a_win_probability += branch_weight
                elif branch_wins_b < 2:
                    next_scenarios.append((
                        _updated_state(
                            state,
                            team_a,
                            team_b,
                            map_name,
                            ratings,
                            actual_a,
                        ),
                        map_order,
                        branch_wins_a,
                        branch_wins_b,
                        branch_weight,
                    ))
        scenarios = next_scenarios

    assert not scenarios, "Every enumerated Bo3 branch must terminate."
    return float(team_a_win_probability)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> dict:
    source_metadata = _load_and_validate_source_metadata(SOURCE_STEM)
    replay_metadata = _load_and_validate_source_metadata(REPLAY_STEM)
    for (_, source), (_, replay) in zip(
        source_metadata, replay_metadata, strict=True
    ):
        source_analytics = source["extra_metadata"]["advanced_analytics"]
        replay_analytics = replay["extra_metadata"]["advanced_analytics"]
        for key in (
            "grand_final_matchups",
            "runner_ups",
            "exact_podiums",
            "cinderella_grand_final_runs",
        ):
            assert source_analytics[key] == replay_analytics[key], (
                f"Replay changed locked aggregate analytics for seed "
                f"{source['seed']}: {key}"
            )
    actual_model_hash = _sha256_file(MODEL_ARTIFACT_PATH)
    assert actual_model_hash == EXPECTED_MODEL_SHA256

    bundle = joblib.load(MODEL_ARTIFACT_PATH)
    assert bundle["feature_columns"] == list(N_ELO_FEATURES_EXPECTED_ORDER)
    predictor = FastIsotonicXGBoostPredictor.from_calibrated_classifier(
        bundle["calibrated_model"]
    )
    starting_state, last_historical_map = build_starting_state()

    opening_probabilities = []
    for team_a, team_b in REPORTED_OPENING_MATCHUPS:
        probability_a = exact_unknown_map_bo3_probability(
            team_a, team_b, starting_state, predictor
        )
        opening_probabilities.append({
            "matchup": (
                f"{TEAM_DISPLAY_NAMES[team_a]} vs {TEAM_DISPLAY_NAMES[team_b]}"
            ),
            "best_of": 3,
            "team_a": TEAM_DISPLAY_NAMES[team_a],
            "team_b": TEAM_DISPLAY_NAMES[team_b],
            "team_a_win_probability": probability_a,
            "team_b_win_probability": 1.0 - probability_a,
        })

    path_schema = replay_metadata[0][1]["extra_metadata"][
        "advanced_analytics"
    ]["complete_path_schema"]
    complete_path_runs = []
    for path, metadata in replay_metadata:
        analytics = metadata["extra_metadata"]["advanced_analytics"]
        top_paths = []
        for rank, item in enumerate(
            analytics["complete_bracket_paths"], start=1
        ):
            teams = item["key"].split("__then__")
            assert len(teams) == len(path_schema)
            top_paths.append({
                "rank": rank,
                "count": item["count"],
                "probability": item["probability"],
                "path": dict(zip(path_schema, teams, strict=True)),
            })
        complete_path_runs.append({
            "seed": metadata["seed"],
            "n_iterations": metadata["n_iterations"],
            "metadata_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "unique_complete_paths": analytics[
                "complete_bracket_path_unique_count"
            ],
            "top_3": top_paths,
        })
    payload = {
        "schema_version": "1.0",
        "tournament_name": "starladder",
        "generated_timestamp_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_runs": [
            {
                "seed": metadata["seed"],
                "n_iterations": metadata["n_iterations"],
                "metadata_path": path.relative_to(PROJECT_ROOT).as_posix(),
            }
            for path, metadata in source_metadata
        ],
        "deterministic_replay_runs": [
            {
                "seed": metadata["seed"],
                "n_iterations": metadata["n_iterations"],
                "metadata_path": path.relative_to(PROJECT_ROOT).as_posix(),
            }
            for path, metadata in replay_metadata
        ],
        "model_artifact_path": MODEL_ARTIFACT_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "model_artifact_sha256": actual_model_hash,
        "elo_snapshot_cutoff": EVENT_START.isoformat(),
        "last_historical_map": last_historical_map.isoformat(),
        "k_factor": K_FACTOR,
        "active_map_pool": list(ACTIVE_MAP_POOL),
        "opening_series_probabilities": {
            "status": "computed_from_locked_model_and_snapshot",
            "method": (
                "exact enumeration of all 210 ordered three-map samples, "
                "including live Elo updates after each map"
            ),
            "historical_vetoes_used": False,
            "results": opening_probabilities,
        },
        "complete_bracket_outcomes": {
            "status": "recovered_by_deterministic_replay",
            "requested_top_k": 3,
            "rerun_performed": True,
            "replay_validation": (
                "All previously published aggregate analytics match the locked "
                "source artifacts exactly for both seeds."
            ),
            "path_schema": path_schema,
            "seed_runs": complete_path_runs,
            "seed_stability_max_absolute_difference": replay_metadata[0][1][
                "extra_metadata"
            ]["seed_stability_max_absolute_differences"][
                "Complete Bracket Path"
            ],
        },
    }
    _atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nExported: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return payload


if __name__ == "__main__":
    main()
