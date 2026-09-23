"""Fit the locked V2 isotonic calibrator on Validation only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_MODULE_DIR = PROJECT_ROOT / "src" / "features"
if str(FEATURE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_MODULE_DIR))

from player_elo_state import (  # noqa: E402
    build_v2_model_feature_frame,
    symmetrize_v2_features,
)


RANDOM_SEED = 42
RAW_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "map_classifier" / "v2_dynamic_hybrid_xgboost.joblib"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "map_classifier" / "v2_dynamic_hybrid_isotonic.joblib"
)
METADATA_PATH = OUTPUT_PATH.with_suffix(".metadata.json")
FEATURE_PATH = PROJECT_ROOT / "data" / "v2_player_team_features.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw_artifact = joblib.load(RAW_ARTIFACT_PATH)
    base_model = raw_artifact["model"]
    feature_columns = list(raw_artifact["feature_columns"])
    player_scale_factor = float(raw_artifact["player_scale_factor"])
    if base_model.get_booster().feature_names != feature_columns:
        raise AssertionError("Raw V2 artifact feature order is inconsistent.")

    # Predicate pushdown is the hard data-access boundary: no Test row is read.
    validation = pd.read_parquet(
        FEATURE_PATH, filters=[("split", "=", "validation")]
    ).sort_values(["datetime", "match_id", "game_id"], kind="stable")
    if len(validation) != 633 or not validation["split"].eq("validation").all():
        raise AssertionError("Expected exactly 633 Validation rows and no other split.")
    if validation["datetime"].max() > pd.Timestamp("2026-03-31 23:59:59"):
        raise AssertionError("Validation exceeded its locked calendar boundary.")

    natural_features = build_v2_model_feature_frame(
        validation, player_scale_factor=player_scale_factor
    ).reindex(columns=feature_columns)
    if natural_features.columns.tolist() != feature_columns:
        raise AssertionError("Validation features do not match the artifact contract.")
    x_validation, y_validation = symmetrize_v2_features(
        natural_features, validation["team1_win"]
    )
    x_validation = x_validation.reindex(columns=feature_columns)

    raw_probability = base_model.predict_proba(x_validation)[:, 1]
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model), method="isotonic"
    )
    calibrated_model.fit(x_validation, y_validation)
    calibrated_probability = calibrated_model.predict_proba(x_validation)[:, 1]

    metrics = {
        "raw_validation_log_loss": float(
            log_loss(y_validation, raw_probability, labels=[0, 1])
        ),
        "raw_validation_brier": float(
            brier_score_loss(y_validation, raw_probability)
        ),
        "calibrated_validation_log_loss": float(
            log_loss(y_validation, calibrated_probability, labels=[0, 1])
        ),
        "calibrated_validation_brier": float(
            brier_score_loss(y_validation, calibrated_probability)
        ),
    }

    bundle = {
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "feature_columns": feature_columns,
        "player_scale_factor": player_scale_factor,
        "seed": RANDOM_SEED,
        "calibration_method": "isotonic",
        "calibration_split": "validation_2026_q1_only",
        "validation_rows_natural": len(validation),
        "validation_rows_symmetrized": len(x_validation),
        "source_artifact_sha256": sha256(RAW_ARTIFACT_PATH),
        "validation_metrics": metrics,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    joblib.dump(bundle, temporary_artifact)
    temporary_artifact.replace(OUTPUT_PATH)
    artifact_hash = sha256(OUTPUT_PATH)

    metadata = {
        "schema_version": "2.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "artifact_sha256": artifact_hash,
        "source_artifact": str(RAW_ARTIFACT_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "source_artifact_sha256": bundle["source_artifact_sha256"],
        "seed": RANDOM_SEED,
        "feature_columns": feature_columns,
        "player_scale_factor": player_scale_factor,
        "calibration_method": "isotonic",
        "calibration_api": "sklearn.frozen.FrozenEstimator",
        "calibration_split": "Validation only: 2026-01-01 through 2026-03-31",
        "validation_rows_natural": len(validation),
        "validation_rows_symmetrized": len(x_validation),
        "test_rows_loaded": 0,
        "metrics": metrics,
    }
    temporary_metadata = METADATA_PATH.with_suffix(METADATA_PATH.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(METADATA_PATH)

    print("Validation-only isotonic calibration complete.")
    print(f"Natural / symmetrized rows: {len(validation)} / {len(x_validation)}")
    print(f"Raw Log-loss: {metrics['raw_validation_log_loss']:.12f}")
    print(f"Raw Brier: {metrics['raw_validation_brier']:.12f}")
    print(f"Calibrated Log-loss: {metrics['calibrated_validation_log_loss']:.12f}")
    print(f"Calibrated Brier: {metrics['calibrated_validation_brier']:.12f}")
    print(f"Artifact SHA-256: {artifact_hash}")
    print("Test rows loaded: 0")


if __name__ == "__main__":
    main()
