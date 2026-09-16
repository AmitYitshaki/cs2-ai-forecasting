# UI & Artifact Export — Architecture Spec

Written for implementation by Codex. Covers two independent pieces: the live-simulation progress UI, and the results export mechanism. Ship them as separate concerns — the exporter wraps `simulate_tournament`, it does not live inside it.

---

## 1. Live Simulation UI (tqdm)

`bracket_simulator.py` already uses `tqdm.auto` correctly ([`bracket_simulator.py:270-291`](../../src/models/bracket_simulator.py#L270-L291)); this section tightens the display, it doesn't replace the mechanism.

**Required bar format**, set explicitly rather than relying on tqdm's default (which differs slightly between plain terminal and notebook widget rendering):

```python
from tqdm.auto import tqdm

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

with tqdm(
    total=n_iterations,
    desc="Tournament Monte Carlo",
    unit="iter",
    bar_format=BAR_FORMAT,
    disable=not show_progress,
) as progress:
    ...
```

- `{rate_fmt}` gives iterations/sec, `{remaining}` gives ETA — both derived automatically by tqdm from elapsed time and progress, no manual calculation needed.
- Keep `disable=not show_progress` — required for headless/scripted runs (e.g. the seed-stability diff, CI) so the bar doesn't spam log output.
- **Do not** add a live-updating postfix (e.g. running leader's `P(Champion)`). It adds complexity and a second place for a bug to hide, for a cosmetic benefit — skip it in favor of robustness, especially under the current timeline pressure.
- Update granularity is already correct via `batch_size` (512–2048) — do not reduce it to update every single iteration; that adds redraw overhead for no benefit at this iteration count.

---

## 2. Artifact Exporter

### Design rule: keep `simulate_tournament` pure

`simulate_tournament` must remain a pure computation function — no disk I/O inside it. The seed-stability diff (two full runs per check) and any other internal/test invocations must not be forced to write files. Build the exporter as a separate wrapper called explicitly after a run is deliberately being persisted.

### Signature

```python
def export_tournament_results(
    summary_df: pd.DataFrame,       # output of print_tournament_summary
    tournament_name: str,           # slug, e.g. "starladder"
    output_root: Path,              # e.g. PROJECT_ROOT / "results"
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
    ...
```

### Directory layout

```
project/results/<tournament_slug>/
    <tournament_slug>_<timestampUTC>_seed<seed>_n<n_iterations>.csv
    <tournament_slug>_<timestampUTC>_seed<seed>_n<n_iterations>.json
    <tournament_slug>_<timestampUTC>_seed<seed>_n<n_iterations>_metadata.json
    runs_index.csv
```

- One subfolder per tournament (`results/starladder/`, `results/pgl_bucharest/`, `results/iem_beijing/`).
- **Timestamp: ISO 8601, UTC, filename-safe** (`20260917T164500Z`) — not local time. A live run during the event itself must not have timezone ambiguity in its own filename.
- **Never overwrite.** Every run (including stability-diff seeds) gets its own timestamped triple of files. Append one row per run to `runs_index.csv` (columns: timestamp, seed, n_iterations, file paths, model artifact hash) so every run stays individually recoverable and the index gives an at-a-glance run history without opening each file.

### File formats

- **CSV**: `summary_df` written as-is (the same table `print_tournament_summary` returns — Team, P(QF), P(SF), P(Final), P(Champion), SE(Champion)).
- **JSON**: same data, `orient="records"`, so it's directly consumable by downstream tooling or a future dashboard without a CSV parser.

### Metadata schema (`..._metadata.json`)

Beyond seed and N, required fields:

```json
{
  "schema_version": "1.0",
  "tournament_name": "starladder",
  "timestamp_utc": "2026-09-17T16:45:00Z",
  "seed": 42,
  "n_iterations": 100000,
  "execution_seconds": 651.2,
  "model_artifact_path": "artifacts/map_classifier/canonical_elo_isotonic.joblib",
  "model_artifact_sha256": "<content hash of the joblib file>",
  "k_factor": 24.0,
  "active_map_pool": ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass"],
  "quarterfinals": [["team_a", "team_b"], ["team_c", "team_d"], ["team_e", "team_f"], ["team_g", "team_h"]],
  "historical_map_orders_used": {"...": "..."},
  "extra_metadata": {}
}
```

- **`model_artifact_sha256`** is required, not optional: it ties a results file to the *exact* model weights used, not just a filename that could be overwritten or refer to a differently-retrained artifact later. Compute via `hashlib.sha256` over the joblib file's bytes.
- **`schema_version`** lets future column/field changes to this export format be handled without silently breaking anything reading these files back later.

### Write discipline

- **Write atomically.** Write each file to a `.tmp` suffix path first, then `os.replace()` (atomic rename) to the final path. A live run interrupted mid-write (real risk during an actual event) must never leave a corrupt or half-written file at the real path.
- Write the metadata file last, after both CSV and JSON have successfully landed — its presence signals "this run's export is complete," useful as a simple completeness check when reviewing `results/<slug>/` later.

### Sequencing note

Per the review findings doc: don't build this against an unverified model artifact. Resolve Finding 1 (locked Test metric vs. deployed model mismatch) before wiring `model_artifact_path`/`model_artifact_sha256` into a live export — there's no value in a polished, hashed, reproducible record of a run made with a model whose real performance hasn't been confirmed.
