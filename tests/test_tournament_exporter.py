"""Unit tests for atomic tournament-result exports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.tournament_exporter import export_tournament_results


class TournamentExporterTests(unittest.TestCase):
    def test_complete_export_and_index(self) -> None:
        summary = pd.DataFrame(
            [{
                "Team": "A", "P(QF)": "100.0%", "P(SF)": "50.0%",
                "P(Final)": "25.0%", "P(Champion)": "12.5%",
                "SE(Champion)": "1.0%",
            }]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.joblib"
            artifact.write_bytes(b"locked model")
            exported = export_tournament_results(
                summary,
                "starladder",
                root / "results",
                42,
                100,
                artifact,
                k_factor=24.0,
                active_map_pool=("a", "b", "c", "d", "e", "f", "g"),
                quarterfinals=(("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")),
                execution_seconds=1.25,
            )
            for path in (
                exported.csv_path,
                exported.json_path,
                exported.metadata_path,
                exported.runs_index_path,
            ):
                self.assertTrue(path.is_file())
            self.assertFalse(list(root.rglob("*.tmp")))
            metadata = json.loads(exported.metadata_path.read_text("utf-8"))
            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(len(metadata["model_artifact_sha256"]), 64)
            index = pd.read_csv(exported.runs_index_path)
            self.assertEqual(len(index), 1)
            self.assertEqual(index.loc[0, "seed"], 42)


if __name__ == "__main__":
    unittest.main()
