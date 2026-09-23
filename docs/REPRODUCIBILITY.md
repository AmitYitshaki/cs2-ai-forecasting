# Reproducibility Guide

This guide separates quick verification from expensive artifact regeneration. You do not need to rerun two million tournaments to review the project.

## Requirements

- Python 3.12
- Git
- The repository's tracked data and artifact files

Create an isolated environment and install the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

## Level 1: fast repository verification

These checks are safe and do not rebuild models, notebooks, or simulations:

```powershell
python -m unittest discover -s tests -v
python -m compileall .
python scripts/check_markdown_links.py
git diff --check
```

The expected unit-test count is 15. The same checks run on pushes and pull requests to `main` through `.github/workflows/python-tests.yml`.

## Level 2: inspect the locked evidence

The authoritative files are already tracked:

- `artifacts/map_classifier/v2_dynamic_hybrid_isotonic.metadata.json`
- `artifacts/map_classifier/v2_dynamic_hybrid_xgboost.metadata.json`
- `results/starladder/starladder_barcelona_v2_final_report.json`
- `results/starladder/starladder_20260922T110617Z_seed42_n1000000_metadata.json`
- `results/starladder/starladder_20260922T110617Z_seed99_n1000000_metadata.json`

Verify the deployed calibrated artifact in PowerShell:

```powershell
(Get-FileHash artifacts/map_classifier/v2_dynamic_hybrid_isotonic.joblib -Algorithm SHA256).Hash.ToLower()
```

Expected value:

```text
a3ab81428571b5147cf944fd14fd7c1aba9e0abbb5b56bf4db12e64a23ebcfd0
```

## Level 3: rebuild derived V2 data and documentation notebooks

The builder scripts are the reproducible source for the V2 notebooks and feature store:

```powershell
python scripts/build_v2_data_exploration_notebook.py
python scripts/build_v2_xgboost_tuning_notebook.py
```

These commands may overwrite their corresponding generated notebooks and parquet outputs. Run them only in a clean branch, then review the diff rather than committing regenerated files blindly.

The key derived tables are:

- `data/v2_player_elo_events.parquet`
- `data/v2_player_team_features.parquet`

## Level 4: rebuild validation-only calibration

```powershell
python scripts/calibrate_v2_model.py
```

Calibration must use the 2026 Q1 Validation split only. The script's metadata must continue to report `test_rows_loaded: 0`.

Rebuilding creates a new artifact identity. Do not replace the published artifact or its hash unless a new version is intentionally being released.

## Level 5: tournament preflight

The default command runs the identity-resolution and replay-parity gates without unlocking the expensive production simulation:

```powershell
python scripts/v2_monte_carlo.py
```

Before a full StarLadder run, confirm:

- all eight teams resolve;
- all 40 configured players resolve;
- there are zero unintended cold starts;
- replay parity is effectively floating-point zero;
- the replayed maximum timestamp is strictly earlier than `2026-09-17T00:00:00`;
- the feature order exactly matches artifact metadata.

## Level 6: full production simulation

```powershell
python scripts/v2_monte_carlo.py --run-full
```

This explicitly unlocks two independent runs of one million complete double-elimination tournaments, using seeds 42 and 99. It is intentionally excluded from CI because it is expensive and produces timestamped artifacts.

The production process must:

1. complete both seeds;
2. satisfy the predefined 0.5 percentage-point stability threshold;
3. atomically export CSV, JSON, and metadata outputs;
4. append both runs to `results/starladder/runs_index.csv`;
5. load the confirmed result only after export lock for post-hoc comparison.

## Determinism and expected variation

- Unit tests and fixed-seed simulations are deterministic for a pinned software environment.
- Runtime varies by hardware and does not affect probabilities.
- Different Monte Carlo seeds should differ slightly; stability is evaluated statistically rather than by expecting byte-identical summaries.
- Rebuilt Joblib artifacts may not be byte-identical across library versions, which is why dependencies and SHA-256 hashes are recorded.

## Test-set policy

The published Test metrics are a one-shot record, not a repeatedly optimized benchmark. Reproduction may verify the locked prediction path, but future model selection must not use the 2026 Q2 Test outcomes. A genuinely new version requires a new forward holdout.
