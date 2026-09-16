# CS2 Tournament Simulator

A leakage-safe forecasting system for professional Counter-Strike 2 tournaments. The project combines a calibrated XGBoost map classifier with point-in-time Elo ratings and Monte Carlo simulation for Bo3/Bo5 series, single-elimination brackets, and eight-team double-elimination brackets.

The repository is intentionally educational: Python identifiers remain conventional English, while every notebook explains the statistical reasoning and engineering decisions in Hebrew.

## What the system does

1. Replays historical maps chronologically and records each team's global and per-map Elo **before** the current result is applied.
2. Splits data strictly by time: Train through 2025-12-31, Validation in 2026 Q1, and the locked Test period from 2026-04-01 onward.
3. Symmetrizes each split independently so the model cannot learn a source-order bias.
4. Trains an XGBoost classifier on six canonical Elo features and fits isotonic calibration on Validation only.
5. Simulates maps, series, and complete tournament brackets while carrying live Elo momentum through each simulated path.
6. Exports timestamped, atomic result artifacts with the random seed, iteration count, runtime, configuration, schema version, and SHA-256 hash of the deployed model.

## Final model

The production classifier uses only:

- `team1_elo_global`, `team2_elo_global`
- `team1_elo_map`, `team2_elo_map`
- `elo_global_diff`, `elo_map_diff`

DNA, head-to-head history, rolling form, and Optuna tuning were each tested through controlled ablations. None cleared the fixed canonical-Elo baseline statistically, so they were deliberately excluded from production rather than stacked onto the model.

The deployed artifact is `artifacts/map_classifier/canonical_elo_isotonic.joblib`. Its current SHA-256 is:

```text
274a8118ada2c9551f7e668f8999fa0ff672b0158b6d055056a152f09820370d
```

## Locked Test result

Notebook 1.5 evaluates the exact calibrated artifact used by the simulator—not a separately retrained approximation.

| Metric | Overall Test | Map 1 | Map 2+ |
|---|---:|---:|---:|
| Accuracy | 0.666387 | 0.641156 | 0.691030 |
| Brier score | 0.214199 | 0.222295 | 0.206291 |
| Log-loss | 0.646857 | — | — |

Bootstrap confidence intervals are resampled by `match_id`, preserving dependence between maps from the same series and between mirrored rows.

## StarLadder production forecast

The final production run simulated StarLadder StarSeries Fall 2026 twice with **1,000,000 tournament paths per seed**.

| Stability distribution | Maximum absolute seed difference |
|---|---:|
| Champion probability | 0.1275% |
| Grand Final matchup | 0.0425% |
| Runner-up | 0.0695% |
| Exact podium | 0.0447% |
| Cinderella Grand Final run | 0.0410% |

All checks cleared the 0.5 percentage-point guardrail. Full-precision results and complete advanced-analytics distributions are stored under `results/starladder/`; see [docs/RESULTS.md](docs/RESULTS.md) for the concise report.

## Notebook walkthrough

| Notebook | Purpose |
|---|---|
| `1.0_setup_and_data_sanity.ipynb` | Environment verification, source-table sanity check, and the first anti-leakage allowlist |
| `1.1_point_in_time_elo_pipeline.ipynb` | Point-in-time global/map Elo and the first Elo-only XGBoost ablation |
| `1.2_xgboost_elo_h2h_dna_candidate.ipynb` | Rejected Elo + H2H + DNA candidate and why more features were not automatically better |
| `1.3_canonical_elo_feature_ablation.ipynb` | Canonical identity fix and controlled DNA, H2H, and rolling-form ablations |
| `1.4_optuna_statistical_guardrail.ipynb` | Optuna tuning and paired bootstrap guardrail against the fixed baseline |
| `1.5_locked_test_artifact_evaluation.ipynb` | One-way locked Test evaluation of the deployed calibrated artifact |
| `1.6_monte_carlo_fast_path_and_cologne.ipynb` | Fast NumPy inference, series simulation, and point-in-time Cologne playoff backtest |
| `1.7_double_elimination_preproduction.ipynb` | 100k-path double-elimination pre-production run and atomic export |
| `1.8_starladder_1m_production.ipynb` | Dual-seed million-path production run and rare-event analytics |

All notebook Markdown is written in Hebrew. The notebooks are executed and retain their outputs so the reasoning and observed results can be reviewed without rerunning long simulations.

## Repository structure

```text
artifacts/                 deployed calibrated model bundle
configs/                   configuration placeholders
data/                      source and derived CS2 tables
docs/                      project brief, work plan, review notes, results
notebooks/                 chronological educational workflow
results/starladder/        immutable timestamped production exports
scripts/                   reproducible audits and production runners
src/data/                  chronological split logic
src/features/              point-in-time Elo/H2H/rolling-form engine
src/models/                series, bracket, fast inference, and export logic
tests/                     unit and pipeline tests
```

## Setup

Python 3.12 is the verified environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To inspect the educational workflow:

```powershell
jupyter lab
```

The million-iteration production run is intentionally expensive. For a smoke test, call the simulator with a much smaller `n_iterations` value rather than rerunning Notebook 1.8 unchanged.

## Reproducibility and safety

- The strict phantom-row filter is `map_name.notna() AND (is_total == False OR bestOf == 1)`, followed by `score1_game + score2_game > 0`.
- Feature construction is chronological; a row is recorded before its outcome updates Elo.
- Symmetrization occurs after the chronological split and independently within each period.
- Test is never used for tuning or calibration.
- The NumPy fast path asserts the exact Booster feature order before inference.
- Tournament state is isolated per Monte Carlo path.
- Result files are written through `.tmp` files and atomically renamed.
- No API keys, passwords, access tokens, private keys, or local absolute paths are required or stored in the repository.

## Data note

The tables under `data/` are derived from public competitive-CS2 datasets and project-specific cleaning/join steps. Their provenance and known limitations are documented in [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md). Users are responsible for observing the terms of the original data sources when redistributing or reusing them.

## Known limitations

- Unknown vetoes are approximated by sampling maps uniformly without replacement from the active pool.
- The model does not explicitly model roster changes, economy, travel, substitutions, or player-level availability.
- Monte Carlo standard error measures simulation noise only; it does not capture model uncertainty or missing information.
- The latest historical map in the StarLadder snapshot is dated 2026-06-21, so later roster/form information is absent.

## Verification

The current suite contains 12 passing tests covering leakage-safe feature construction, chronological splitting, series simulation, single- and double-elimination invariants, analytics reconciliation, and atomic exports.

For the detailed development history and statistical conclusions, follow the notebooks in numeric order and read [docs/RESULTS.md](docs/RESULTS.md).
