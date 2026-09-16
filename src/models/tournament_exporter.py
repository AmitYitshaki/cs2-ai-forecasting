"""Atomic, provenance-rich exports for tournament simulation results."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "1.0"
INDEX_COLUMNS = (
    "timestamp_utc",
    "tournament_name",
    "seed",
    "n_iterations",
    "csv_path",
    "json_path",
    "metadata_path",
    "model_artifact_sha256",
)


@dataclass(frozen=True)
class ExportedArtifacts:
    """Paths produced by one complete atomic export."""

    csv_path: Path
    json_path: Path
    metadata_path: Path
    runs_index_path: Path


def export_tournament_results(
    summary_df: pd.DataFrame,
    tournament_name: str,
    output_root: Path,
    seed: int,
    n_iterations: int,
    model_artifact_path: Path,
    *,
    k_factor: float,
    active_map_pool: Sequence[str],
    quarterfinals: Sequence[tuple[str, str]],
    historical_map_orders: Mapping[frozenset[str], Sequence[str]] | None = None,
    execution_seconds: float | None = None,
    extra_metadata: dict | None = None,
) -> ExportedArtifacts:
    """Persist a complete run without exposing partially written final files."""

    slug = _validate_slug(tournament_name)
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    artifact = Path(model_artifact_path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact}")
    _validate_summary_schema(summary_df)

    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    timestamp_file = timestamp.strftime("%Y%m%dT%H%M%SZ")
    timestamp_iso = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    tournament_dir = Path(output_root).resolve() / slug
    tournament_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slug}_{timestamp_file}_seed{seed}_n{n_iterations}"
    csv_path = tournament_dir / f"{stem}.csv"
    json_path = tournament_dir / f"{stem}.json"
    metadata_path = tournament_dir / f"{stem}_metadata.json"
    runs_index_path = tournament_dir / "runs_index.csv"
    for path in (csv_path, json_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing export: {path}")

    artifact_hash = _sha256_file(artifact)
    project_root = Path(output_root).resolve().parent
    try:
        artifact_display = artifact.relative_to(project_root).as_posix()
    except ValueError:
        artifact_display = str(artifact)
    serialized_orders = {
        "__vs__".join(sorted(map(str, matchup))): list(order)
        for matchup, order in (historical_map_orders or {}).items()
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tournament_name": slug,
        "timestamp_utc": timestamp_iso,
        "seed": int(seed),
        "n_iterations": int(n_iterations),
        "execution_seconds": (
            None if execution_seconds is None else float(execution_seconds)
        ),
        "model_artifact_path": artifact_display,
        "model_artifact_sha256": artifact_hash,
        "k_factor": float(k_factor),
        "active_map_pool": list(active_map_pool),
        "quarterfinals": [list(pairing) for pairing in quarterfinals],
        "historical_map_orders_used": serialized_orders,
        "extra_metadata": extra_metadata or {},
    }

    _atomic_dataframe_csv(summary_df, csv_path)
    _atomic_dataframe_json(summary_df, json_path)
    # Metadata lands last and therefore acts as the run-completeness marker.
    _atomic_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    _append_index_atomically(
        runs_index_path,
        {
            "timestamp_utc": timestamp_iso,
            "tournament_name": slug,
            "seed": int(seed),
            "n_iterations": int(n_iterations),
            "csv_path": csv_path.relative_to(project_root).as_posix(),
            "json_path": json_path.relative_to(project_root).as_posix(),
            "metadata_path": metadata_path.relative_to(project_root).as_posix(),
            "model_artifact_sha256": artifact_hash,
        },
    )
    return ExportedArtifacts(csv_path, json_path, metadata_path, runs_index_path)


def _validate_slug(value: str) -> str:
    slug = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]+", slug):
        raise ValueError(
            "tournament_name must be a safe slug containing a-z, 0-9, _ or -."
        )
    return slug


def _validate_summary_schema(summary_df: pd.DataFrame) -> None:
    expected = [
        "Team", "P(QF)", "P(SF)", "P(Final)", "P(Champion)", "SE(Champion)"
    ]
    if list(summary_df.columns) != expected:
        raise ValueError(f"Unexpected summary schema: {list(summary_df.columns)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_suffix(final_path.suffix + ".tmp")


def _atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_dataframe_json(frame: pd.DataFrame, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_json(
            temporary, orient="records", indent=2, force_ascii=False
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_index_atomically(path: Path, row: dict[str, object]) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(INDEX_COLUMNS):
                raise ValueError("Existing runs_index.csv has an incompatible schema.")
            existing.extend(reader)

    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(row)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
