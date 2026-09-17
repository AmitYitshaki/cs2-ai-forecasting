# Version 2.0 Architecture Plan

## Objective

Version 2.0 will extend the leakage-safe canonical Elo system with player-level micro-analytics and explicit time decay. This document is an implementation plan, not a finalized modeling specification. Every feature must be computable from information available before the predicted map begins.

## Planned data sources

| Source | Candidate fields | Temporal requirement |
|---|---|---|
| Existing map and player tables | Player IDs, team IDs, kills, deaths, assists, ADR, KAST, map, event, timestamp | Use only maps completed before the prediction timestamp |
| HLTV player pages or licensed extracts | HLTV rating, Rating 2.0/2.1 components, event rating, map rating | Store source timestamp and rating definition; never backfill a later rating into an earlier match |
| Roster history | Team membership, join/leave dates, substitutions | Resolve the active five-player roster at the prediction timestamp |
| Tournament context | Event tier, LAN/online flag, opponent strength, best-of format | Must be known before the series starts |

External-source ingestion will require a provenance record containing the source URL or dataset identifier, retrieval time, coverage period, schema version, and applicable reuse terms.

## Point-in-time player feature store

The player feature store will be keyed by canonical `player_id`, `team_id`, `map_name`, and an effective timestamp. A training row may join only to the latest feature record whose effective timestamp is strictly earlier than the map timestamp.

Planned pre-match features include:

- exponentially weighted ADR, KAST, kills per round, deaths per round, and HLTV rating;
- global and per-map player form;
- number of prior maps and effective sample size;
- roster continuity and days played together;
- team aggregates and within-roster dispersion;
- explicit missingness and cold-start indicators.

Post-match statistics from the current map remain forbidden inputs.

## Time decay

For an observation recorded at time `t_i` and a prediction at time `t`, its unnormalized recency weight is

```text
w_i(t) = exp(-lambda * (t - t_i))
```

Using a half-life `h` expressed in days,

```text
lambda = ln(2) / h
w_i(t) = 2 ** (-(t - t_i) / h)
```

The exponentially weighted mean of a player statistic `x` is

```text
EWMean_t(x) = sum_i(w_i(t) * x_i) / sum_i(w_i(t))
```

Candidate half-lives will be selected on the chronological validation period only. Train, validation, and test boundaries remain unchanged from Version 1.0.

## Player-to-team aggregation

Let `s_p(t)` be a point-in-time player strength measure and `q_p(t)` its reliability weight. For the active roster `R_T(t)`, the team-level player signal is

```text
PlayerStrength_T(t) = sum_{p in R_T(t)} q_p(t) * s_p(t)
                      / sum_{p in R_T(t)} q_p(t)
```

Reliability may be based on effective sample size, with shrinkage toward a population prior:

```text
ShrunkStat_p = (n_eff,p * PlayerStat_p + alpha * PopulationPrior)
               / (n_eff,p + alpha)
```

Candidate team features include the roster mean, median, minimum, maximum, standard deviation, top-player contribution, and weakest-player contribution. Signed team differences will be created only after the chronological split and will flip under symmetrization.

## Elo integration candidates

The Version 1.0 Elo expectation remains

```text
E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
```

Two integration strategies will be evaluated independently:

1. Keep Elo unchanged and provide player aggregates to the calibrated map classifier.
2. Add a bounded pre-match roster adjustment to the Elo difference:

```text
Delta_adjusted = (R_A - R_B) + beta * (PlayerStrength_A - PlayerStrength_B)
E_A = 1 / (1 + 10 ** (-Delta_adjusted / 400))
```

The coefficient `beta`, shrinkage strength `alpha`, and decay half-life `h` must be selected without touching the locked test period.

## Leakage and identity constraints

- Resolve canonical team and player identities before feature construction.
- Sort deterministically by `datetime`, `match_id`, and `game_id`.
- Emit every player and Elo feature before applying the current map result.
- Apply the phantom-row filter before building history.
- Split chronologically before symmetrization or learned preprocessing.
- Fit calibration on validation only.
- Preserve explicit cold-start flags rather than silently substituting mature estimates.
- Audit roster joins for duplicates, overlapping memberships, and future-effective records.

## Evaluation plan

Version 2.0 candidates will be compared with the deployed canonical-Elo baseline through controlled ablations:

1. Elo plus time decay only.
2. Elo plus ADR-derived player features.
3. Elo plus HLTV rating features.
4. Elo plus player features and time decay.

Primary metrics are log-loss and Brier score. Accuracy, calibration curves, Map 1 versus Map 2+ strata, roster-continuity strata, and match-cluster bootstrap confidence intervals will remain secondary diagnostics. A new model advances only if it clears the predefined statistical guardrail on validation.

## Initial deliverables

- Data-source and licensing audit.
- Canonical player identity table and roster timeline.
- Point-in-time player feature builder with leakage tests.
- Time-decay implementation with configurable half-life.
- Coverage and cold-start report.
- Ablation notebook and validation comparison against Version 1.0.
