"""V2 Dynamic Hybrid Elo StarLadder Barcelona cutoff and identity pre-flight.

This command intentionally stops at the Phase-4 confirmation gate. It does not
run the million-iteration tournament until a later, explicit user instruction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_MODULE_DIR = PROJECT_ROOT / "src" / "features"
for import_path in (PROJECT_ROOT, FEATURE_MODULE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.build_v2_data_exploration_notebook import (  # noqa: E402
    normalize_player,
    normalize_team,
)
from player_elo_state import (  # noqa: E402
    K_PERF,
    K_TEAM,
    PLAYER_HALF_LIFE_DAYS,
    TEAM_HALF_LIFE_DAYS,
    PlayerEloReplayEngine,
    PlayerEloTournamentState,
)
from src.models.bracket_simulator import (  # noqa: E402
    FastNamedIsotonicXGBoostPredictor,
    TournamentAnalytics,
    print_tournament_summary,
    simulate_double_elimination_tournament,
)
from src.models.tournament_exporter import export_tournament_results  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "starladder_barcelona_rosters.json"
PRIMARY_PATH = PROJECT_ROOT / "data" / "final_tournament_features.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "v2_player_team_features.parquet"
EVENT_PATH = PROJECT_ROOT / "data" / "v2_player_elo_events.parquet"
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "map_classifier" / "v2_dynamic_hybrid_isotonic.joblib"
)
RESOLUTION_REPORT_PATH = (
    PROJECT_ROOT / "results" / "starladder" / "starladder_barcelona_v2_resolution.json"
)
LOCKED_AS_OF_DATE = pd.Timestamp("2026-09-17T00:00:00")
FULL_RUN_ITERATIONS = 1_000_000
INVALID_ROSTER_DIAGNOSTIC_CAP = 1_000
SEEDS = (42, 99)
PREFLIGHT_ITERATIONS = 5_000
BATCH_SIZE = 5_000
GRAND_FINAL_RESET = False
ACTIVE_MAP_POOL = (
    "Cache", "Dust2", "Mirage", "Inferno", "Nuke", "Ancient", "Anubis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iterations", type=int, default=FULL_RUN_ITERATIONS)
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Pass the explicit Phase-4 gate and execute both production seeds.",
    )
    return parser.parse_args()


def _pre_event_json_text(path: Path) -> str:
    """Remove the top-level outcome member before structured JSON parsing.

    The outcome object contains only scalar strings by contract. This strict
    expression fails closed if it cannot remove exactly one member.
    """

    raw_text = path.read_text(encoding="utf-8")
    quoted_string = r'"(?:\\.|[^"\\])*"'
    scalar_object_body = rf'(?:\s*{quoted_string}\s*:\s*{quoted_string}\s*,?)*'
    pattern = re.compile(
        rf'\s*"confirmed_result"\s*:\s*\{{{scalar_object_body}\}}\s*,',
        flags=re.DOTALL,
    )
    sanitized, replacements = pattern.subn("", raw_text, count=1)
    if replacements != 1 or '"confirmed_result"' in sanitized:
        raise RuntimeError("Could not isolate confirmed_result from pre-event config.")
    return sanitized


def load_pre_event_config(path: Path) -> dict[str, Any]:
    """Parse rosters/seeding without materializing the known outcome object."""

    payload = json.loads(_pre_event_json_text(path))
    if "confirmed_result" in payload:
        raise AssertionError("Post-event outcome entered the pre-event config.")
    return payload


def load_confirmed_result_posthoc(path: Path) -> dict[str, Any]:
    """Load the known outcome only after a future simulation has completed."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("confirmed_result")
    if not isinstance(result, dict):
        raise ValueError("confirmed_result is missing from the post-hoc config.")
    return result


def validate_rosters(
    config: dict[str, Any], requested_iterations: int
) -> tuple[dict[str, dict[str, Any]], int]:
    teams = config.get("teams")
    errors: list[str] = []
    if not isinstance(teams, dict) or len(teams) != 8:
        errors.append(f"config.teams: expected 8 teams, received {type(teams).__name__}")
        teams = teams if isinstance(teams, dict) else {}
    for team_name, team_payload in teams.items():
        if not isinstance(team_payload, dict):
            errors.append(f"{team_name}: team record is not an object")
            continue
        if team_payload.get("roster_verified") is not True:
            errors.append(f"{team_name}.roster_verified: expected true")
        players = team_payload.get("players")
        if not isinstance(players, list):
            errors.append(f"{team_name}.players: expected a 5-player array")
            continue
        for slot in range(5):
            value = players[slot] if slot < len(players) else None
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{team_name}.players[{slot}]: missing player")
        if len(players) > 5:
            for slot in range(5, len(players)):
                errors.append(f"{team_name}.players[{slot}]: unexpected extra player")

    if errors:
        print("ROSTER VERIFICATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        capped = min(requested_iterations, INVALID_ROSTER_DIAGNOSTIC_CAP)
        print(f"Simulation cap enforced: N={capped:,}")
        raise SystemExit(
            "Refusing to construct tournament features until every listed slot is verified."
        )
    print("Roster verification passed: 8 teams, 40 players, all attested verified.")
    return teams, requested_iterations


def load_replay_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = pd.read_parquet(FEATURE_PATH).sort_values(
        ["datetime", "match_id", "game_id"], kind="stable"
    ).reset_index(drop=True)
    maps["_row_id"] = np.arange(len(maps), dtype=np.int64)

    events = pd.read_parquet(EVENT_PATH).sort_values(
        ["_row_id", "side", "slot"], kind="stable"
    ).reset_index(drop=True)
    raw = pd.read_csv(PRIMARY_PATH, low_memory=False)
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="raise")
    is_total = raw["is_total"].fillna(False).astype(bool)
    best_of_one = pd.to_numeric(raw["bestOf"], errors="coerce").eq(1)
    score_sum = pd.to_numeric(raw["score1_game"], errors="coerce") + pd.to_numeric(
        raw["score2_game"], errors="coerce"
    )
    valid = raw["map_name"].notna() & (~is_total | best_of_one) & score_sum.gt(0)
    identity_maps = raw.loc[valid].sort_values(
        ["datetime", "match_id", "game_id"], kind="stable"
    ).reset_index(drop=True)
    if len(identity_maps) != len(maps):
        raise AssertionError("Primary identity rows do not align with V2 map features.")
    if not np.array_equal(
        identity_maps["match_id"].to_numpy(), maps["match_id"].to_numpy()
    ):
        raise AssertionError("Match identity order drifted from the training pipeline.")

    identity_rows: list[dict[str, Any]] = []
    for row_id, row in identity_maps.iterrows():
        for side in (1, 2):
            for slot in range(1, 6):
                prefix = f"team{side}_player{slot}"
                player_id = row.get(f"{prefix}_id")
                player_name = row.get(prefix)
                if pd.notna(player_id):
                    player_key = f"id:{int(float(player_id))}"
                elif normalize_player(player_name):
                    player_key = f"name:{normalize_player(player_name)}"
                else:
                    player_key = f"unknown:{row['match_id']}:{side}:{slot}"
                identity_rows.append(
                    {
                        "_row_id": row_id,
                        "side": side,
                        "slot": slot,
                        "identity_player_key": player_key,
                        "player_name": player_name,
                        "player_norm": normalize_player(player_name),
                    }
                )
    identities = pd.DataFrame(identity_rows)
    events = events.merge(
        identities, on=["_row_id", "side", "slot"], how="left", validate="one_to_one"
    )
    if not events["player_key"].eq(events["identity_player_key"]).all():
        raise AssertionError("Historical player identity keys do not reproduce training.")
    events = events.drop(columns="identity_player_key")
    return maps, events


def resolution_report(
    engine: PlayerEloReplayEngine,
    teams: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    resolved_teams: dict[str, str] = {}
    resolved_rosters: dict[str, list[str]] = {}
    for input_team, payload in teams.items():
        team_norm = normalize_team(input_team)
        team_key = engine.resolve_team(team_norm)
        resolved_teams[input_team] = team_key
        team_status = engine.team_status(team_key)
        team_rows.append(
            {
                "input_name": input_team,
                "normalized_name": team_norm,
                **team_status,
            }
        )
        resolved_rosters[input_team] = []
        for slot, input_player in enumerate(payload["players"], start=1):
            player_norm = normalize_player(input_player)
            player_key = engine.resolve_player(player_norm)
            resolved_rosters[input_team].append(player_key)
            status = engine.player_status(player_key)
            player_rows.append(
                {
                    "team": input_team,
                    "slot": slot,
                    "input_name": input_player,
                    "normalized_name": player_norm,
                    **status,
                }
            )
    return (
        pd.DataFrame(team_rows),
        pd.DataFrame(player_rows),
        resolved_teams,
        resolved_rosters,
    )


def assert_replay_matches_locked_features(
    replayed: pd.DataFrame, locked: pd.DataFrame
) -> float:
    """Prove the extracted engine reproduces the previously locked state."""

    audit_columns = [
        "team1_elo_decay", "team2_elo_decay",
        "team1_gap_days", "team2_gap_days",
        "team1_player_agg_elo", "team2_player_agg_elo",
        "team1_player_elo_decay_std", "team2_player_elo_decay_std",
        "lineup_prior_maps_diff",
        "team1_player_cold_starts", "team2_player_cold_starts",
    ]
    reference = locked.loc[
        locked["_row_id"].isin(replayed["_row_id"]), ["_row_id", *audit_columns]
    ].sort_values("_row_id", kind="stable")
    candidate = replayed[["_row_id", *audit_columns]].sort_values(
        "_row_id", kind="stable"
    )
    if not np.array_equal(reference["_row_id"], candidate["_row_id"]):
        raise AssertionError("Replay audit row identities do not align.")
    differences = np.abs(
        reference[audit_columns].to_numpy(dtype=float)
        - candidate[audit_columns].to_numpy(dtype=float)
    )
    maximum = float(np.nanmax(differences))
    if not np.allclose(
        reference[audit_columns].to_numpy(dtype=float),
        candidate[audit_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    ):
        raise AssertionError(f"Extracted replay engine drifted; max difference={maximum}.")
    return maximum


def export_resolution_report(
    *,
    replay: Any,
    team_report: pd.DataFrame,
    player_report: pd.DataFrame,
    opening_features: pd.DataFrame,
    feature_columns: list[str],
    replay_max_difference: float,
) -> None:
    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        output = frame.copy()
        if "last_seen" in output:
            output["last_seen"] = output["last_seen"].map(
                lambda value: None if pd.isna(value) else pd.Timestamp(value).isoformat()
            )
        return output.to_dict(orient="records")

    payload = {
        "schema_version": "2.0.0-resolution-gate",
        "as_of_date_exclusive": replay.as_of_date.isoformat(),
        "replayed_rows": replay.replayed_rows,
        "last_replayed_timestamp": replay.last_replayed_timestamp.isoformat()
        if replay.last_replayed_timestamp is not None
        else None,
        "replay_parity_max_absolute_difference": replay_max_difference,
        "team_half_life_days": TEAM_HALF_LIFE_DAYS,
        "player_half_life_days": PLAYER_HALF_LIFE_DAYS,
        "k_perf": K_PERF,
        "calibrated_artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "calibrated_artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "feature_columns": feature_columns,
        "teams": records(team_report),
        "players": records(player_report),
        "opening_matchup_diagnostics": records(opening_features),
        "cold_start_player_count": int(player_report["cold_start"].sum()),
        "confirmed_result_loaded": False,
        "simulation_run": False,
    }
    RESOLUTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESOLUTION_REPORT_PATH.with_suffix(
        RESOLUTION_REPORT_PATH.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(RESOLUTION_REPORT_PATH)


class SlowNamedPredictor:
    """Pandas/scikit-learn reference path used only in the 5k pre-flight."""

    def __init__(self, calibrated_model: object, feature_columns: list[str]) -> None:
        self.calibrated_model = calibrated_model
        self.feature_columns = feature_columns

    def predict_numpy(self, matrix: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(matrix, columns=self.feature_columns)
        return np.asarray(self.calibrated_model.predict_proba(frame)[:, 1], dtype=float)


def run_tournament(
    opening_matchups: list[tuple[str, str]],
    starting_state: PlayerEloTournamentState,
    predictor: object,
    *,
    n_iterations: int,
    seed: int,
    show_progress: bool,
) -> tuple[dict[str, dict[str, int]], TournamentAnalytics, float]:
    started = perf_counter()
    counts, analytics = simulate_double_elimination_tournament(
        opening_matchups,
        starting_state,
        predictor,
        ACTIVE_MAP_POOL,
        n_iterations=n_iterations,
        random_state=seed,
        batch_size=min(BATCH_SIZE, n_iterations),
        grand_final_reset=GRAND_FINAL_RESET,
        track_analytics=True,
        show_progress=show_progress,
    )
    return counts, analytics, perf_counter() - started


def assert_fast_slow_preflight(
    opening_matchups: list[tuple[str, str]],
    starting_state: PlayerEloTournamentState,
    fast_predictor: object,
    calibrated_model: object,
    feature_columns: list[str],
) -> dict[str, float]:
    fast_counts, fast_analytics, fast_seconds = run_tournament(
        opening_matchups,
        starting_state,
        fast_predictor,
        n_iterations=PREFLIGHT_ITERATIONS,
        seed=314_159,
        show_progress=False,
    )
    slow_counts, slow_analytics, slow_seconds = run_tournament(
        opening_matchups,
        starting_state,
        SlowNamedPredictor(calibrated_model, feature_columns),
        n_iterations=PREFLIGHT_ITERATIONS,
        seed=314_159,
        show_progress=False,
    )
    assert fast_counts == slow_counts
    for field_name in (
        "grand_final_matchups",
        "runner_ups",
        "exact_podiums",
        "grand_final_appearances",
        "complete_paths",
        "bracket_paths",
    ):
        assert getattr(fast_analytics, field_name) == getattr(
            slow_analytics, field_name
        )
    print(
        "5k fast/slow pre-flight passed exactly: "
        f"fast={fast_seconds:.3f}s, slow={slow_seconds:.3f}s"
    )
    return {"fast_seconds": fast_seconds, "slow_seconds": slow_seconds}


def max_counter_probability_difference(
    first: Counter, second: Counter, n_iterations: int
) -> float:
    keys = set(first) | set(second)
    return max(
        (abs(first[key] - second[key]) / n_iterations for key in keys),
        default=0.0,
    )


def analytics_metadata(
    analytics: TournamentAnalytics,
    counts: dict[str, dict[str, int]],
    n_iterations: int,
) -> dict[str, Any]:
    champions = Counter({team: values["Champion"] for team, values in counts.items()})
    cinderella = analytics.cinderella_runs(champions, n_iterations)

    def serialize(counter: Counter, labeler, limit: int | None = None) -> list[dict]:
        return [
            {
                "key": labeler(key),
                "count": int(count),
                "probability": count / n_iterations,
            }
            for key, count in counter.most_common(limit)
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
        "cinderella_definition": "P(Champion) < 0.10 in this V2 run",
        "cinderella_grand_final_runs": serialize(cinderella, str),
        "most_probable_brackets": serialize(
            analytics.bracket_paths, str, limit=5
        ),
        "bracket_path_unique_count": len(analytics.bracket_paths),
    }


def _actual_gf_path_match(path: str, champion: str, runner_up: str) -> bool:
    fields = dict(part.split(":", 1) for part in path.split("|"))
    final_winner = fields.get("GrandFinalReset", fields.get("GrandFinal"))
    finalists = {fields.get("UB_Final"), fields.get("LB_Final")}
    return final_winner == champion and finalists == {champion, runner_up}


def posthoc_comparison(
    confirmed_result: dict[str, Any],
    counts: dict[str, dict[str, int]],
    analytics: TournamentAnalytics,
    n_iterations: int,
) -> dict[str, Any]:
    champion = str(confirmed_result["champion"])
    runner_up = str(confirmed_result["runner_up"])
    champion_order = sorted(
        counts, key=lambda team: counts[team]["Champion"], reverse=True
    )
    top_five = analytics.bracket_paths.most_common(5)
    matching_top_five = [
        {"rank": rank, "path": path, "probability": count / n_iterations}
        for rank, (path, count) in enumerate(top_five, start=1)
        if _actual_gf_path_match(path, champion, runner_up)
    ]
    best_matching: dict[str, Any] | None = None
    for rank, (path, count) in enumerate(
        analytics.bracket_paths.most_common(), start=1
    ):
        if _actual_gf_path_match(path, champion, runner_up):
            best_matching = {
                "rank": rank,
                "path": path,
                "probability": count / n_iterations,
            }
            break
    return {
        "loaded_after_exports_locked": True,
        "confirmed_result": confirmed_result,
        "champion_probability": counts[champion]["Champion"] / n_iterations,
        "champion_rank": champion_order.index(champion) + 1,
        "top_five_contains_actual_gf_outcome": bool(matching_top_five),
        "matching_top_five_paths": matching_top_five,
        "highest_ranked_actual_gf_outcome_path": best_matching,
        "scoreline_tracking_note": (
            "The MPB contract tracks match winners, not Grand Final scorelines; "
            "the confirmed 3-1 is therefore descriptive only."
        ),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()

    # This is deliberately the first stateful operation in the command.
    pre_event_config = load_pre_event_config(CONFIG_PATH)
    teams, effective_iterations = validate_rosters(
        pre_event_config, args.n_iterations
    )
    as_of_date = pd.Timestamp(pre_event_config["as_of_date"])
    if as_of_date != LOCKED_AS_OF_DATE:
        raise AssertionError(
            f"as_of_date must equal {LOCKED_AS_OF_DATE.isoformat()}, received {as_of_date}."
        )

    model_artifact = joblib.load(MODEL_PATH)
    feature_columns = list(model_artifact["feature_columns"])
    player_scale_factor = float(model_artifact["player_scale_factor"])
    calibrated_model = model_artifact["calibrated_model"]

    maps, events = load_replay_inputs()
    engine = PlayerEloReplayEngine(
        maps,
        events,
        team_half_life_days=TEAM_HALF_LIFE_DAYS,
        player_half_life_days=PLAYER_HALF_LIFE_DAYS,
        k_perf=K_PERF,
        team_normalizer=normalize_team,
        player_normalizer=normalize_player,
    )
    replay = engine.replay_until(as_of_date)
    if replay.last_replayed_timestamp is not None and replay.last_replayed_timestamp >= as_of_date:
        raise AssertionError("Cutoff violation: replay included an event at/after as_of_date.")
    if engine.replayed_timestamps_ and max(engine.replayed_timestamps_) >= as_of_date:
        raise AssertionError("Cutoff violation in replay timestamp audit.")
    replay_max_difference = assert_replay_matches_locked_features(
        replay.features, maps
    )

    team_report, player_report, resolved_teams, resolved_rosters = resolution_report(
        engine, teams
    )
    if len(team_report) != 8 or len(player_report) != 40:
        raise AssertionError("Resolution report must contain 8 teams and 40 players.")

    feature_rows: list[dict[str, Any]] = []
    for team1_name, team2_name in pre_event_config["opening_matchups"]:
        feature_mapping = engine.build_matchup_feature_dict(
            resolved_teams[team1_name],
            resolved_teams[team2_name],
            resolved_rosters[team1_name],
            resolved_rosters[team2_name],
            at=as_of_date,
            player_scale_factor=player_scale_factor,
        )
        ordered = pd.DataFrame([feature_mapping]).reindex(columns=feature_columns)
        if ordered.columns.tolist() != feature_columns:
            raise AssertionError("Feature reindexing failed the stored model contract.")
        if ordered.isna().any().any():
            raise AssertionError("Opening-match feature vector contains NaN.")
        probability = float(calibrated_model.predict_proba(ordered)[0, 1])
        feature_rows.append(
            {
                "team1": team1_name,
                "team2": team2_name,
                "calibrated_team1_win_probability": probability,
                **feature_mapping,
            }
        )
    opening_features = pd.DataFrame(feature_rows)
    export_resolution_report(
        replay=replay,
        team_report=team_report,
        player_report=player_report,
        opening_features=opening_features,
        feature_columns=feature_columns,
        replay_max_difference=replay_max_difference,
    )

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    print(f"Hard cutoff: {as_of_date.isoformat()} (exclusive)")
    print(f"Replayed maps: {replay.replayed_rows:,}")
    print(f"Last replayed timestamp: {replay.last_replayed_timestamp}")
    print(f"Replay parity max absolute difference: {replay_max_difference:.3e}")
    print(f"Locked parameters: team_half_life={TEAM_HALF_LIFE_DAYS:g}, player_half_life={PLAYER_HALF_LIFE_DAYS:g}, K_perf={K_PERF:g}")
    print(f"Stored feature order verified: {feature_columns}")
    print("\nOpening-match diagnostic features/probabilities:")
    print(opening_features.to_string(index=False))
    print("\nTEAM RESOLUTION REPORT (8/8):")
    print(team_report.to_string(index=False))
    print("\nPLAYER RESOLUTION REPORT (40/40):")
    print(player_report.to_string(index=False))

    cold_players = player_report.loc[player_report["cold_start"]]
    print("\nCOLD-START SUMMARY:")
    if cold_players.empty:
        print("  No player cold starts. All 40 players have prior maps.")
    else:
        print(f"  !!! {len(cold_players)} PLAYER COLD START(S) DETECTED !!!")
        print(
            cold_players[["team", "slot", "input_name", "canonical_key"]].to_string(
                index=False
            )
        )
    print(f"Confirmed result loaded into structured memory: {False}")
    print(f"Resolution artifact: {RESOLUTION_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Requested future full-run size: {effective_iterations:,}")
    if not args.run_full:
        print("STOP: Phase-4 confirmation gate reached. No tournament simulation was run.")
        return

    if not cold_players.empty:
        raise RuntimeError("Full production run blocked by player cold starts.")
    if team_report["cold_start"].any():
        raise RuntimeError("Full production run blocked by team cold starts.")
    if replay_max_difference != 0.0:
        raise RuntimeError("Full production run blocked by replay parity drift.")
    if effective_iterations != FULL_RUN_ITERATIONS:
        raise ValueError(
            f"Production execution requires exactly {FULL_RUN_ITERATIONS:,} iterations."
        )

    opening_matchups = [
        (str(team1), str(team2))
        for team1, team2 in pre_event_config["opening_matchups"]
    ]
    starting_state = PlayerEloTournamentState.from_replay_engine(
        engine,
        resolved_teams=resolved_teams,
        resolved_rosters=resolved_rosters,
        at=as_of_date,
        player_scale_factor=player_scale_factor,
    )
    fast_predictor = FastNamedIsotonicXGBoostPredictor.from_calibrated_classifier(
        calibrated_model, feature_columns
    )
    # The named mapping and the NumPy matrix must agree before any Monte Carlo call.
    opening_matrix = np.asarray(
        [
            tuple(starting_state.build_matchup_feature_dict(a, b).values())
            for a, b in opening_matchups
        ],
        dtype=np.float32,
    )
    slow_opening = np.asarray(
        calibrated_model.predict_proba(
            pd.DataFrame(opening_matrix, columns=feature_columns)
        )[:, 1],
        dtype=float,
    )
    fast_opening = fast_predictor.predict_numpy(opening_matrix)
    inference_max_difference = float(np.max(np.abs(fast_opening - slow_opening)))
    if not np.allclose(fast_opening, slow_opening, rtol=0.0, atol=1e-7):
        raise AssertionError(
            f"Fast inference parity failed: max difference={inference_max_difference}."
        )
    preflight_timings = assert_fast_slow_preflight(
        opening_matchups,
        starting_state,
        fast_predictor,
        calibrated_model,
        feature_columns,
    )

    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        counts, analytics, seconds = run_tournament(
            opening_matchups,
            starting_state,
            fast_predictor,
            n_iterations=FULL_RUN_ITERATIONS,
            seed=seed,
            show_progress=True,
        )
        print(f"seed={seed} execution_seconds={seconds:.3f}")
        summary = print_tournament_summary(counts, FULL_RUN_ITERATIONS)
        runs[seed] = {
            "counts": counts,
            "analytics": analytics,
            "seconds": seconds,
            "summary": summary,
        }

    counts_42 = runs[42]["counts"]
    counts_99 = runs[99]["counts"]
    analytics_42 = runs[42]["analytics"]
    analytics_99 = runs[99]["analytics"]
    champion_42 = Counter(
        {team: values["Champion"] for team, values in counts_42.items()}
    )
    champion_99 = Counter(
        {team: values["Champion"] for team, values in counts_99.items()}
    )
    cinderella_42 = analytics_42.cinderella_runs(
        champion_42, FULL_RUN_ITERATIONS
    )
    cinderella_99 = analytics_99.cinderella_runs(
        champion_99, FULL_RUN_ITERATIONS
    )
    stability = {
        "P(Champion)": max_counter_probability_difference(
            champion_42, champion_99, FULL_RUN_ITERATIONS
        ),
        "Grand Final Matchup": max_counter_probability_difference(
            analytics_42.grand_final_matchups,
            analytics_99.grand_final_matchups,
            FULL_RUN_ITERATIONS,
        ),
        "Runner-Up": max_counter_probability_difference(
            analytics_42.runner_ups,
            analytics_99.runner_ups,
            FULL_RUN_ITERATIONS,
        ),
        "Exact Podium": max_counter_probability_difference(
            analytics_42.exact_podiums,
            analytics_99.exact_podiums,
            FULL_RUN_ITERATIONS,
        ),
        "Cinderella Grand Final": max_counter_probability_difference(
            cinderella_42, cinderella_99, FULL_RUN_ITERATIONS
        ),
    }
    for name, difference in stability.items():
        print(f"{name}: max absolute seed difference={difference:.4%}")
    assert max(stability.values()) <= 0.005, stability
    seed42_top_path = analytics_42.bracket_paths.most_common(1)[0][0]
    seed99_top3_paths = {
        path for path, _ in analytics_99.bracket_paths.most_common(3)
    }
    mpb_top1_in_seed99_top3 = seed42_top_path in seed99_top3_paths
    print(
        "MPB stability: seed-42 top-1 path appears in seed-99 top-3 = "
        f"{mpb_top1_in_seed99_top3}"
    )

    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    production_checks = {
        "replay_max_difference": replay_max_difference,
        "fast_slow_opening_probability_max_difference": inference_max_difference,
        "fast_slow_5k_exact_match": True,
        "preflight_timings_seconds": preflight_timings,
        "resolved_teams": int((~team_report["cold_start"]).sum()),
        "resolved_players": int((~player_report["cold_start"]).sum()),
        "cold_start_players": int(player_report["cold_start"].sum()),
        "calibration_artifact_sha256": model_hash,
        "as_of_date_exclusive": as_of_date.isoformat(),
    }
    for seed in SEEDS:
        exported = export_tournament_results(
            runs[seed]["summary"],
            "starladder",
            PROJECT_ROOT / "results",
            seed,
            FULL_RUN_ITERATIONS,
            MODEL_PATH,
            k_factor=K_TEAM,
            active_map_pool=ACTIVE_MAP_POOL,
            quarterfinals=opening_matchups,
            historical_map_orders={},
            execution_seconds=runs[seed]["seconds"],
            extra_metadata={
                "model_version": "V2 Dynamic Hybrid Elo",
                "format": "eight_team_double_elimination",
                "grand_final_best_of": 5,
                "other_matches_best_of": 3,
                "grand_final_reset": GRAND_FINAL_RESET,
                "cutoff_enforcement": "exclusive",
                "production_checks": production_checks,
                "seed_stability_max_absolute_differences": stability,
                "mpb_seed42_top1_in_seed99_top3": mpb_top1_in_seed99_top3,
                "advanced_analytics": analytics_metadata(
                    runs[seed]["analytics"],
                    runs[seed]["counts"],
                    FULL_RUN_ITERATIONS,
                ),
                "confirmed_result_loaded": False,
            },
        )
        runs[seed]["exported"] = exported
        print(
            f"seed={seed} export locked: "
            f"{exported.metadata_path.relative_to(PROJECT_ROOT)}"
        )

    # One-way provenance gate: the outcome is materialized only after both
    # simulation exports and their metadata completeness markers exist.
    if not all(runs[seed].get("exported") for seed in SEEDS):
        raise AssertionError("Post-hoc outcome gate opened before exports locked.")
    confirmed_result = load_confirmed_result_posthoc(CONFIG_PATH)
    posthoc = posthoc_comparison(
        confirmed_result, counts_42, analytics_42, FULL_RUN_ITERATIONS
    )
    top_five = [
        {"rank": rank, "path": path, "count": int(count), "probability": count / FULL_RUN_ITERATIONS}
        for rank, (path, count) in enumerate(
            analytics_42.bracket_paths.most_common(5), start=1
        )
    ]
    p1, p2 = top_five[0]["probability"], top_five[1]["probability"]
    conservative_se = float(
        np.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / FULL_RUN_ITERATIONS)
    )
    mpb_standout = bool((p1 - p2) > 1.96 * conservative_se)
    final_report = {
        "schema_version": "2.0.0-starladder-final-report",
        "n_iterations_per_seed": FULL_RUN_ITERATIONS,
        "primary_seed": 42,
        "model_artifact_sha256": model_hash,
        "production_checks": production_checks,
        "seed_stability": stability,
        "mpb_seed42_top1_in_seed99_top3": mpb_top1_in_seed99_top3,
        "seed42_champion_probabilities": {
            team: values["Champion"] / FULL_RUN_ITERATIONS
            for team, values in sorted(
                counts_42.items(), key=lambda item: item[1]["Champion"], reverse=True
            )
        },
        "seed42_advanced_analytics": analytics_metadata(
            analytics_42, counts_42, FULL_RUN_ITERATIONS
        ),
        "seed42_top5_most_probable_brackets": top_five,
        "mpb_top1_clear_standout_95pct_mc_rule": mpb_standout,
        "mpb_top1_minus_top2": p1 - p2,
        "posthoc_comparison": posthoc,
        "exports": {
            str(seed): {
                "csv": str(runs[seed]["exported"].csv_path.relative_to(PROJECT_ROOT)),
                "json": str(runs[seed]["exported"].json_path.relative_to(PROJECT_ROOT)),
                "metadata": str(runs[seed]["exported"].metadata_path.relative_to(PROJECT_ROOT)),
            }
            for seed in SEEDS
        },
    }
    report_path = (
        PROJECT_ROOT / "results" / "starladder" /
        "starladder_barcelona_v2_final_report.json"
    )
    atomic_json(report_path, final_report)
    print(f"Post-hoc result loaded after exports: {confirmed_result}")
    print(f"Final report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
