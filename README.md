# CS2 Dynamic Hybrid Elo Tournament Forecaster

**Can a model forecast an entire professional Counter-Strike 2 tournament without seeing the future?** This project replays every historical map point-in-time, combines organization-level and player-level Elo, calibrates an XGBoost map classifier, and simulates complete Bo3/Bo5 double-elimination brackets millions of times.

Version 2.0 is a portfolio-grade, leakage-controlled forecasting system: reproducible inputs, explicit temporal boundaries, validation-only model decisions, a one-shot locked Test evaluation, fast NumPy inference, and atomic result exports with provenance hashes.

## Why Version 2.0 exists

Version 1.0 modeled only the team: global Elo, map-specific Elo, and their differences. That macro view was competitive, but it had a structural blind spot. When an organization changed its lineup, the model continued to value the team through results earned by players who might no longer be on the server.

Version 2.0 adds the missing micro layer: persistent player ratings tied to the verified active five. The model now sees both the strength of the organization and the strength of the people representing it.

## Dynamic Hybrid Elo

The production pipeline replays valid maps in strict chronological order. Every feature is emitted **before** the current result updates state.

```text
Historical maps
    -> point-in-time replay
    -> Macro Elo: team identity, 180-day half-life
    -> Micro Elo: mean rating of the active five, 1,095-day half-life
    -> variance matching fitted on Train only
    -> nine named model features
    -> fixed XGBoost classifier
    -> isotonic calibration fitted on Validation only
    -> Bo3/Bo5 series simulation
    -> double-elimination Monte Carlo bracket
```

The two rating layers decay differently because they represent different things:

- **Team Elo half-life: 180 days.** Systems, coaching, roles, and lineup chemistry can become stale quickly.
- **Player Elo half-life: 1,095 days.** Individual skill behaved as a much more persistent signal in the validation sweep.
- **Player aggregation: mean of the active five.** A mean preserves the Elo scale; a sum would inflate it fivefold.

The player update began with a centered performance term:

```text
player_delta = team_delta / 5 + K_PERF * relative_performance
```

After 400 validation-grid combinations, the winning value was **`K_PERF = 0`**. In this model, CS2 is about winning, not padding stats: map-level ADR/KAST/K-D adjustments added noise beyond the team result. That is an empirical project finding, not a universal claim that player statistics never matter.

## Production feature contract

The locked classifier consumes exactly nine features, reindexed by stored name before inference:

1. `BaseElo_diff`
2. `PlayerAggElo_diff_scaled`
3. `team1_gap_days`
4. `team2_gap_days`
5. `player_elo_std_A_scaled`
6. `player_elo_std_B_scaled`
7. `lineup_prior_maps_diff`
8. `team1_player_cold_starts`
9. `team2_player_cold_starts`

The deployed calibrated artifact is [`v2_dynamic_hybrid_isotonic.joblib`](artifacts/map_classifier/v2_dynamic_hybrid_isotonic.joblib). Its SHA-256 is:

```text
a3ab81428571b5147cf944fd14fd7c1aba9e0abbb5b56bf4db12e64a23ebcfd0
```

## Statistical discipline: the model we rejected

A 75-trial Optuna search produced a slightly better validation point estimate than the fixed baseline. A paired, match-cluster bootstrap told a different story:

```text
95% CI for tuned-minus-untuned per-row log-loss: [-0.0042, +0.0005]
```

Because the interval crossed zero, the apparent gain could not be distinguished from sampling noise. The tuned model was rejected and the fixed, untuned learned-blend model was locked for Test. Hyperparameter search was treated as a hypothesis, not permission to publish the lowest number.

## Locked Test results

The Test partition contains 595 map rows from 2026 Q2 and was evaluated exactly once after all feature and model choices were locked.

| Model | Accuracy | Log-loss | Brier score |
|---|---:|---:|---:|
| V1 canonical team Elo | 66.64% | 0.6469 | 0.2142 |
| **V2 Dynamic Hybrid Elo** | **68.07%** | **0.6099** | **0.2100** |

The full IEM Cologne 2026 slice contains 187 rows: 68.98% accuracy, 0.6081 log-loss, and 0.2092 Brier score.

## StarLadder Barcelona: one million brackets per seed

The retrospective StarLadder StarSeries Fall 2026 backtest used verified rosters for all 40 players and an exclusive information cutoff of **2026-09-17 00:00:00**. The actual result was inaccessible to feature construction and simulation, and was loaded only after both exported runs were locked.

| Team | P(Champion), seed 42 |
|---|---:|
| **Vitality** | **38.49%** |
| FURIA | 20.37% |
| Natus Vincere | 15.62% |
| MOUZ | 10.73% |
| Aurora | 9.46% |
| magic | 3.82% |
| MIBR | 1.44% |
| NRG | 0.07% |

Vitality was the model's clear favorite and won the tournament. Aurora began below the 10% title threshold, yet the simulator assigned it a **21.93% Cinderella probability of reaching the Grand Final**. Aurora did exactly that before losing to Vitality 3-1.

Two independent runs of 1,000,000 complete brackets used seeds 42 and 99. The largest seed disagreement across tracked headline categories was **0.1319 percentage points**, comfortably inside the predeclared 0.5-point stability guardrail.

Detailed probabilities, rare-event analytics, complete bracket paths, and provenance live in [`results/starladder/`](results/starladder/) and [`docs/RESULTS.md`](docs/RESULTS.md). Prior V1 simulations are preserved separately under [`results/archive/v1/`](results/archive/v1/).

## Leakage controls and reproducibility

- Invalid map rows are removed with `map_name.notna() AND (is_total == False OR bestOf == 1)`, followed by `score1_game + score2_game > 0`.
- Calendar splits are fixed: Train through 2025-12-31, Validation in 2026 Q1, and locked Test in 2026 Q2.
- Stateful features are captured before updating on the current map.
- Scaling is fitted on Train only; calibration is fitted on Validation only.
- Test is excluded from feature selection, hyperparameter tuning, and calibration.
- StarLadder replay enforces a hard exclusive pre-event timestamp.
- All team and player identities must resolve; unintended cold starts block production.
- The fast predictor asserts the exact stored feature order before NumPy inference.
- Tournament state is isolated per Monte Carlo path.
- Results use temporary files followed by atomic rename, with seed, runtime, schema version, and model SHA-256 recorded in metadata.

## Repository guide

```text
artifacts/map_classifier/  locked raw and calibrated model bundles + metadata
config/                    verified tournament rosters and bracket inputs
data/                      source tables and derived point-in-time feature stores
docs/                      architecture, decisions, results, and post-mortem
notebooks/                 bilingual educational experiments and evaluations
results/starladder/        final V2 million-path production exports
results/archive/v1/        preserved V1 simulation runs
scripts/                   reproducible builders, calibration, and production runners
src/features/              point-in-time Elo and tabular feature engineering
src/models/                calibrated inference, series/bracket simulation, exporters
tests/                     15 automated tests for invariants and leakage controls
```

The notebooks present English explanations first and preserve the original Hebrew teaching text immediately afterward.

## Reproduce the project

Python 3.12 is the verified environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Key entry points:

```powershell
# Build the V2 point-in-time feature store and educational notebook
python scripts/build_v2_data_exploration_notebook.py

# Rebuild the validation/tuning notebook
python scripts/build_v2_xgboost_tuning_notebook.py

# Fit validation-only isotonic calibration for the locked raw classifier
python scripts/calibrate_v2_model.py

# Run identity resolution, replay parity, and the small preflight only
python scripts/v2_monte_carlo.py

# Explicitly unlock the expensive dual-seed production simulation
python scripts/v2_monte_carlo.py --run-full
```

The last command runs two million complete tournament simulations in total. Existing immutable exports should be inspected before rerunning it.

## Tech stack

- **Data:** pandas, NumPy, PyArrow
- **Modeling:** XGBoost, scikit-learn isotonic calibration, Optuna
- **Evaluation:** log-loss, Brier score, accuracy, match-cluster bootstrap confidence intervals
- **Simulation:** NumPy fast path, pure-Python state threading, `tqdm` progress reporting
- **Artifacts:** Joblib, JSON/CSV metadata, SHA-256 provenance, atomic writes
- **Quality:** `unittest`, deterministic seeds, explicit runtime assertions

The current suite contains **15 tests** covering filtering, chronology, replay parity, feature ordering, series behavior, single- and double-elimination invariants, analytics reconciliation, and artifact export.

## Limitations

- This is a retrospective research project, not betting or financial advice.
- The StarLadder result validates one tournament backtest; it does not establish universal out-of-sample dominance.
- Elo captures historical outcomes and identity continuity, not tactical matchups, travel, health, substitutions announced after the cutoff, or every roster-role interaction.
- `K_PERF = 0` means the tested box-score adjustment failed here; richer genuinely point-in-time player data could still add value.
- Unknown vetoes and map orders require assumptions, which introduce structural uncertainty.
- Monte Carlo standard error measures simulation noise, not model misspecification or missing data.
- The 1,095-day player decay was the locked validation choice, but the no-decay boundary probe was close and not independently bootstrapped.

## Further reading

- [`docs/V2_ARCHITECTURE_PLAN.md`](docs/V2_ARCHITECTURE_PLAN.md) — mathematical design and leakage constraints
- [`docs/V2_EXECUTIVE_SUMMARY.md`](docs/V2_EXECUTIVE_SUMMARY.md) — the V2 story and real-world validation
- [`docs/V2_POST_MORTEM.md`](docs/V2_POST_MORTEM.md) — accepted and rejected hypotheses
- [`docs/RESULTS.md`](docs/RESULTS.md) — canonical metrics and simulation tables
- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) — original project scope and data notes
