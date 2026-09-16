"""Build immutable chronological datasets for map-classifier modeling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.splitter import ChronologicalSplitConfig, ChronologicalSplitter
from src.features import MapDatasetPreparer, SymmetricFeatureEngineer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "final_tournament_features.csv"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data"
OUTPUT_FILENAMES = {
    "train": "train.parquet",
    "val": "val.parquet",
    "test": "test.parquet",
}


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one file without loading it at once."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write a parquet file completely before replacing its destination."""

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_parquet(temporary_path, index=False, engine="pyarrow")
    temporary_path.replace(output_path)


def build_locked_splits(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Filter, split, independently engineer, and persist all datasets."""

    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_directory / filename
        for name, filename in OUTPUT_FILENAMES.items()
    }
    manifest_path = output_directory / "split_manifest.json"
    protected_paths = [*output_paths.values(), manifest_path]
    existing_paths = [path for path in protected_paths if path.exists()]
    if existing_paths and not force:
        existing_text = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Locked artifacts already exist: {existing_text}. Use --force to replace."
        )

    raw_df = pd.read_csv(input_path, low_memory=False)
    preparer = MapDatasetPreparer()
    prepared_df, filter_audit = preparer.transform(raw_df)

    split_config = ChronologicalSplitConfig()
    splitter = ChronologicalSplitter(split_config)
    base_splits = splitter.split(prepared_df)

    feature_engineer = SymmetricFeatureEngineer()
    engineered_splits = {
        name: feature_engineer.transform(split_df)
        for name, split_df in base_splits.items()
    }

    for name, engineered_df in engineered_splits.items():
        write_parquet_atomically(engineered_df, output_paths[name])

    artifacts = {
        name: {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "rows_before_symmetrization": len(base_splits[name]),
            "rows_after_symmetrization": len(engineered_splits[name]),
            "columns": len(engineered_splits[name].columns),
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
    }
    manifest = {
        "source": {
            "path": str(input_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(input_path),
        },
        "filter": {
            "rule": "map_name.notna() AND (is_total == False OR bestOf == 1)",
            "score_sanity_rule": "score1_game + score2_game > 0",
            **filter_audit.to_dict(),
        },
        "split_boundaries": {
            "train": "datetime < 2026-01-01",
            "val": "2026-01-01 <= datetime < 2026-04-01",
            "test": "datetime >= 2026-04-01",
        },
        "ordering": [
            "filter_and_allowlist",
            "chronological_split",
            "independent_symmetrization_and_diff_features",
        ],
        "artifacts": artifacts,
    }
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")
    temporary_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest_path.replace(manifest_path)
    return manifest


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the locked-split build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to final_tournament_features.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for locked parquet artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing locked artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the production dataset build and print its auditable counts."""

    arguments = parse_arguments()
    manifest = build_locked_splits(
        input_path=arguments.input.resolve(),
        output_directory=arguments.output_directory.resolve(),
        force=arguments.force,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
