# Version 2.0 Architecture Plan

> **Status: implemented, with evidence-driven changes.** This file is the original design record, so some sections intentionally describe hypotheses that were later rejected. The deployed architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md), final metrics in [RESULTS.md](RESULTS.md), and accepted/rejected decisions in [V2_POST_MORTEM.md](V2_POST_MORTEM.md).

## Locked implementation outcomes

| Design question | Production decision |
|---|---|
| Team inactivity decay | 180-day half-life |
| Player inactivity decay | 1,095-day half-life |
| Within-map performance adjustment | Rejected; `K_PERF = 0` |
| Player aggregation | Mean of the active five |
| Hand-engineered blend | Lost to the learned two-feature comparison |
| Production model | Untuned Approach B: separate `BaseElo_diff` and scaled `PlayerAggElo_diff` plus seven reliability/context features |
| Optuna candidate | Rejected because the paired bootstrap CI crossed zero |
| Calibration | Isotonic, fitted on Validation only |

Sections below remain useful for understanding the hypotheses and safeguards, but this table and `ARCHITECTURE.md` describe what actually shipped.

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

### Data availability audit — what's already in-house versus genuinely new

The table above understates how much of this already exists in `data/`. Auditing before scoping acquisition work, in the same style as the Version 1.0 Feature Factory audit:

| Need | Status | Detail |
|---|---|---|
| Per-player identity, per match | ✅ Already have | `final_tournament_features.csv` carries `team{1,2}_player{1-5}_id` and per-player kills/deaths — sufficient to build `lineup_key` and `games_together` with no new sourcing. |
| Core per-player box score (kills, deaths, assists, ADR, KAST) | ✅ Already have, full timeline, corrected finding | `final_tournament_features.csv` (sourced from `cs2_tier1_games.csv`) carries `team{1,2}_player{1-5}_{kills,deaths,assists,adr,kast,kddiff}` — confirmed present end-to-end from 2023-01 through 2026-06. This is **not** subject to the October 2025 cliff: it is the correct primary source for `s_p(t)` across the entire Train/Val/Test range, superseding the earlier, more cautious note about this source (below). |
| Enrichment-only per-player box score (Rating, KPR, DPR, opening kills, clutches) | ⚠️ Have it, coverage-capped, enrichment only | `cs2_newestcombinedmatches.csv` has `team{N}_player_{i}_{RATING,ADR,KAST,KPR,DPR}` plus opening-kill/clutch context, but stops at 2025-10-16 — the same cliff that limited the Version 1.0 rolling-form ablation. **Per Lead directive, this data is used, not discarded**: it enriches Player Elo's quality for any match before the cliff, and its presence is tracked as an explicit `has_rich_stats` flag per player-map so the model can learn to trust post-cliff estimates less if warranted. It is never the sole or primary source, since it cannot cover Validation or Test. |
| Player role / position labels | ❌ Genuinely missing | Not present in any current file. This is the one item in this document that requires real new sourcing (an HLTV-profile scrape or a manually maintained mapping) — and per the role-weighting decision above, it's explicitly out of scope for Version 2.0, so this does not block the initial build. |
| Roster join/leave dates | 🔜 Derivable, not required | Exact-lineup continuity (above) can be computed directly from existing per-match lineup columns by diffing consecutive matches — a separate roster-transactions feed is not required for Version 2.0 and should not be scoped as a blocking dependency. |

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

### Rating-level decay (distinct from feature-level decay above)

The EWMA above solves recency for *derived stat features* (ADR, KAST, etc.), each computed fresh from a window of historical rows. It does not solve recency for the Elo rating itself: Elo is a stateful recursion (`global_ratings_[team]`), not a windowed statistic, so a team that hasn't played in months keeps the exact rating from its last game, with no explicit growth in uncertainty during the gap. This is the literal version of "a 2023 match weighted the same as last week" that motivated this plan.

Apply a second, independent decay directly to the stored rating, evaluated at the start of processing each team's next match using the elapsed gap since their previous one:

```text
R_effective(t) = mu + (R_stored - mu) * 2 ** (-gap_days / h_rating)
```

where `mu` is the neutral prior (1500) and `h_rating` is a separate half-life from the feature-level EWMA above — expected to be longer (core skill erodes slower than short-term form), tuned independently on validation. For teams with normal weekly cadence, `gap_days` is small and this is a no-op; for a team returning from a multi-month gap (off-season, Major break), it correctly regresses their frozen rating toward the mean before it's used as a pre-match feature.

A complementary, optional mechanism: temporarily inflate the K-factor for the first game back after a long gap, so the rating re-converges faster once fresh evidence arrives (standard practice in Glicko-family systems via rating-deviation growth):

```text
K_effective = K_base * (1 + beta * min(gap_days / gap_saturation_days, 1))
```

Both `R_effective` and `K_effective` should be implemented as an extension of the existing `PointInTimeEngine`, not a parallel system — the sequential replay, canonical identity resolution, and phantom-row filtering it already provides are directly reusable, and duplicating that machinery would risk exactly the kind of identity/leakage bugs this project has already spent significant effort finding and fixing in Version 1.0.

## Player Elo: the update rule (Lead directive — Performance-Adjusted Player Elo)

The existing draft referenced a player strength measure `s_p(t)` without defining how it is actually computed from a match. This is the missing piece, and it is also where the "use all available data, don't discard the approach at the cliff" directive is implemented mathematically.

Player Elo cannot use the standard two-player Elo update directly — CS2 box scores don't attribute a map win or loss to one player, only to the team. Instead, each map's **team-level** Elo delta (computed exactly as in Version 1.0) is redistributed across the five active players based on their individual performance that map, so a player who "hard carries" gains more in a win and **loses less in a loss** — the delta is decomposed into a flat baseline plus a performance adjustment centered on the team's own average that map, not a share-based multiplier of the team delta:

```text
team_delta = K_team * (actual_result - expected_team_win_prob)          # unchanged from V1

Impact_p   = z(ADR_p) + z(KAST_p) + z(kill_death_diff_p)                # per-map, per-player
mean_impact_side = mean(Impact_p for p in the 5 active players, this map)
relative_performance_p = Impact_p - mean_impact_side                     # centered: + above own team's average, - below

player_delta_p = (team_delta / 5) + K_perf * relative_performance_p
PlayerElo_p(t+) = PlayerElo_p(t-) + player_delta_p
```

**This is a correction to an earlier implementation**, worth being explicit about: a share-multiplicative version (`player_delta_p = team_delta * 5 * contribution_p`, `contribution_p` always non-negative) was built first and works correctly on a win, but on a loss it makes the strongest individual performer lose *the most* — multiplying a negative `team_delta` by a larger positive share makes it more negative, not less. The centered form above fixes this: `relative_performance_p` sums to zero across the five players by construction, so `sum_p(player_delta_p) = team_delta` exactly regardless of win or loss, and the sign of the adjustment is tied to relative performance rather than flipping with the team's result. `K_perf` (Elo points per standard deviation of relative impact) is a new hyperparameter, selected on validation, not hardcoded.

`Impact_p` is deliberately built only from `ADR`, `KAST`, and kill/death differential — the columns confirmed available with full 2023–2026 coverage above — so the core update rule never hits the cliff. `z(...)` is a per-map z-score against that map's ten participating players, keeping the scale stable across patches and meta shifts without needing a fixed external reference. This is an internal Impact Score, not a claim to reproduce HLTV's (proprietary, undisclosed) Rating formula.

**Where the October 2025 data is used, not discarded:** for any map with `has_rich_stats = true` (i.e. covered by `cs2_newestcombinedmatches.csv`), blend its Rating/KPR/DPR/opening-kill/clutch fields into `Impact_p` as an additional, more informative term before normalizing — a strictly higher-fidelity `Impact_p` for the ~2023–2025-10 window, gracefully degrading to the ADR/KAST/KD version afterward rather than being unavailable. The player-level time decay below (not the data cliff) is what correctly discounts a player's confidence the longer they go without a fresh update of *either* kind — this is the actual mechanism that makes "use everything up to the cliff" safe rather than stale: a player's rating from a rich pre-cliff update doesn't freeze with false confidence into 2026, it decays toward the prior at the same rate as any other aging observation.

Cold start (a player with no prior maps) follows the same convention as everywhere else in this codebase: initialize at the population prior (1500) with `n_eff = 0`, no imputation, explicit reliability flag.

### Player-level time decay (Lead directive — recent matches must dominate)

The stored `PlayerElo_p` is decayed with the identical rating-level mechanism already specified above for Team Elo, applied per player instead of per team:

```text
PlayerElo_effective_p(t) = mu + (PlayerElo_p - mu) * 2 ** (-gap_days_p / h_player)
```

`gap_days_p` is the elapsed time since player `p`'s own most recent recorded map, regardless of which team they were on at the time — since this is keyed by `player_id`, not `team_id`, a transfer between teams doesn't interrupt continuity, only genuine inactivity does. `h_player` is its own hyperparameter, expected a priori to be **shorter** than `h_rating` (team-level): individual mechanical form is understood to be more volatile match-to-match than a team's overall system or identity, but this is a hypothesis to confirm on validation, not an assumption to hardcode.

## Player-to-team aggregation

Let `s_p(t) = PlayerElo_effective_p(t)` (defined below, after decay) be the point-in-time player strength measure and `q_p(t)` its reliability weight. For the active roster `R_T(t)`, the team-level player signal is

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

### Role weighting: excluded by Lead directive — locked decision, with one precision needed

**Locked: all five players are treated as mathematically equal contributors in the team aggregate.** No role-label source exists in the current data (`data/`), and inferring role heuristically (e.g., from AWP-kill share) would be unvalidated guesswork indistinguishable from a modeling artifact later. `PlayerStrength_T(t)` is therefore an **unweighted, equal-share mean** across the active five:

```text
PlayerStrength_T(t) = (1/5) * sum_{p in R_T(t)} s_p(t)
```

replacing the `q_p(t)`-weighted mean above — `q_p(t)` is set uniformly, not by role.

**One distinction worth keeping explicit, so this isn't misread as "ignore data reliability too":** equal weighting applies at the *aggregation* step (5 players, one share each — a role/positional judgment, correctly excluded). It does not apply at the *individual rating estimation* step — a player with 3 career maps and a player with 300 still need the shrinkage-toward-prior treatment (`ShrunkStat_p` above) before their `s_p(t)` enters the equal-weight average, or a single small-sample outlier could swing a fifth of the team's aggregate rating on noise alone. This is a statistical correction for sample size, not a role weight, and it stays in scope.

The minimum/weakest-player-contribution feature remains in the candidate feature list as a role-agnostic proxy for roster fragility — it mirrors `weakest_link_advantage`, a feature already present in `cs2_newestcombinedmatches.csv`, so the concept has precedent in this project's own data, without requiring role labels.

### Roster continuity: needs an explicit formula, not just a candidate feature

"Roster continuity and days played together" is listed above as a feature but has no defined computation. Propose a synergy multiplier applied to the aggregated player signal:

```text
synergy_factor(t) = synergy_floor + (1 - synergy_floor) * min(games_together(t) / synergy_saturation_games, 1)
TeamRating_T(t) = mu + (PlayerStrength_T(t) - mu) * synergy_factor(t)
```

`games_together(t)` counts maps played by the *exact* current five-player lineup, keyed by a sorted tuple of the five active player IDs, reset to zero on any single roster change (V2.0 baseline — a partial-continuity version that discounts rather than resets on a single substitution is a natural V2.1 refinement, not required for the first cut). `synergy_floor` and `synergy_saturation_games` are additional hyperparameters to select on validation, alongside `alpha`, `beta`, and `h`.

A brand-new lineup, or a player with no prior history, starts at the population prior with `n_eff = 0` — same cold-start convention as canonical team Elo elsewhere in this codebase: no imputation, an explicit reliability/cold-start flag, and native handling by the downstream model.

## Dynamic Hybrid Elo (Lead directive — a unified rating engineered before the model)

The Version 1.0 Elo expectation remains

```text
E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
```

This section supersedes the earlier "Golden Path" approach of feeding Team Elo and Player Elo to XGBoost as separate feature pairs and letting the trees learn the interaction. Per Lead directive, Version 2.0 instead engineers a single blended rating before the data reaches the model.

### Team Base Elo

Unchanged from Version 1.0: a traditional Elo tracking the team/org entity strictly from match outcomes, insensitive to exactly which five players are on the server. Call this `BaseElo_T(t)`.

### Aggregating the active five into a team-scale rating

The five players' individual (decayed, performance-adjusted) ratings are combined by **mean, not sum**:

```text
PlayerAggElo_T(t) = (1/5) * sum_{p in active 5} PlayerElo_effective_p(t)
```

This is a deliberate correction worth being explicit about: summing five ratings each centered near 1500 produces a value near 7500, which is not on a comparable scale to `BaseElo_T(t)` (~1500) — blending a ~1500 quantity against a ~7500 quantity would make any blend weight mean something other than what's intended, and would break the standard `/400` logistic formula below. Averaging keeps both terms on the same 1500-centered scale.

### The blend

```text
HybridElo_T(t) = w * BaseElo_T(t) + (1 - w) * PlayerAggElo_T(t)              # same w applied to both teams
P(A wins) = 1 / (1 + 10 ** ((HybridElo_B(t) - HybridElo_A(t)) / 400))
```

**Before trusting this blend, verify the two terms are actually on comparable spread, not just comparable center.** `BaseElo` and `PlayerAggElo` are updated by related but distinct mechanisms (different effective K dynamics, different decay half-lives), so even though both are centered at 1500, their standard deviation across the dataset is not guaranteed to match. Compare `std(BaseElo_diff)` against `std(PlayerAggElo_diff)` on Train before sweeping `w`; if they diverge meaningfully, standardize both to comparable spread first, or `w` will not behave as the interpretable "how much weight" knob it's intended to be — whichever term has higher variance will dominate regardless of the nominal weight.

### Roster-transfer readiness

This falls out of the blend directly, not as a separate mechanism: `BaseElo_T(t)` only moves when the organization's own matches resolve, while `PlayerAggElo_T(t)` recomputes fresh from whichever five players are active on a given matchday — so a transfer shifts it immediately, before the new lineup has played a single match under the org's banner. `w` is the dial controlling how much a single transfer can move the team's blended rating ahead of new results confirming it: a smaller `w` reacts faster to a transfer, a larger `w` stays skeptical until the new lineup proves itself.

### Selecting `w`: a pre-model sweep, not an XGBoost hyperparameter

This is pure arithmetic — no model training required, so it can be swept finely and cheaply, before the classifier is ever touched:

1. Grid `w` from 0.00 to 1.00 in steps of 0.01 (101 points). For each, compute `P(A wins)` directly from the formula above and score Validation log-loss and Brier.
2. **Guard against the same small-Validation multiple-comparisons risk already encountered with Optuna trial counts and the seed-stability checks in this project**: scoring 633 raw Validation rows 101 times risks selecting noise as "optimal." After finding `w* = argmin`, bootstrap (cluster-by-match, the same method used for the Version 1.0 Test confidence interval) a CI around its log-loss and compare against a neutral reference (`w = 0.5`). Only adopt `w*` if it clears that reference outside the CI; default to `w = 0.5` otherwise rather than lock in an unproven decimal.
3. Report `w*` separately for Map 1 versus Map 2+ as a secondary diagnostic — if the optimal blend differs by stratum, that's a real finding worth surfacing even if one locked `w` ships.
4. Lock `w*`, then compute `HybridElo_A`, `HybridElo_B`, and `HybridElo_diff` as the primary macro+micro strength features for XGBoost.

**Keep the two-feature alternative as a comparison ablation, not a deletion.** Feed `TeamElo_diff` and `PlayerAggElo_diff` to XGBoost separately as a second candidate feature set, and compare against the single locked `HybridElo_diff` on Validation. Every prior hand-designed feature decision in this project that wasn't ablated against an alternative (static DNA rates, head-to-head counts, rolling form) turned out to add noise rather than signal once actually tested — there's no reason to assume the hand-blended version is automatically better without checking, even though it's now the primary design.

The weight `w` (once locked), `K_perf`, shrinkage strength `alpha`, synergy parameters, and both decay half-lives (`h_rating`, `h_player`) must all be selected without touching the locked test period.

## Chronological split (Lead directive — strict 80/10/10)

The directive specifies a strict chronological 80% Train / 10% Validation / 10% Test split. Worth confirming against what's already running in production rather than assuming a conflict: the existing calendar-anchored boundaries (Train ≤ 2025-12, Validation = 2026 Q1, Test = 2026 Q2) were checked against the current valid-map row counts and land at **81.7% / 9.45% / 8.88%** — already a close match to 80/10/10 in practice, not a different scheme.

**Recommendation: keep the calendar anchors rather than recomputing a literal percentage-based cut**, for a reason specific to this dataset: Test's boundary was deliberately chosen so the entire IEM Cologne Major 2026 — every map of it — falls inside the holdout as one deliberate real-event anchor, not split across a boundary. A literal row-count-based 80/10/10 cut, recomputed after Version 2.0's feature set changes row counts (player cold-start filtering, roster-join dedup), could shift the Test start date by days or weeks and risks slicing a real tournament's own matches across the Val/Test line — which would be a strictly worse property than a split that is already ~2 percentage points off an exact 80/10/10 target. If the directive means something more literal than this — an exact percentage cut regardless of which events land where — that should be confirmed explicitly before implementation, since it's a change to a boundary this project has treated as load-bearing throughout Version 1.0.

Whichever boundary is used, the one-shot Test discipline from Version 1.0 is unchanged: Test is touched exactly once, for the final selected Version 2.0 candidate only.

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

1. Elo plus rating-level time decay only (no player layer).
2. Elo plus Player Elo (performance-adjusted `Impact_p`, no rich-stats enrichment, no decay) — isolates the core, full-coverage player mechanism on its own.
3. Elo plus Player Elo with rich-stats enrichment for the pre-cliff window added.
4. Elo plus Player Elo with player-level time decay added.
5. Dynamic Hybrid Elo: `HybridElo_diff` at the swept-and-guardrailed `w*`, fed to XGBoost as the primary macro+micro feature.
6. Comparison candidate: `TeamElo_diff` and `PlayerAggElo_diff` fed to XGBoost as two separate features, evaluated against step 5 on Validation.

Each step isolates exactly one new mechanism against the previous, continuing the single-variable-at-a-time discipline that caught DNA, head-to-head, and rolling form adding noise rather than signal in Version 1.0 — Dynamic Hybrid Elo is the second-to-last step, not the first thing tested, and step 6 exists specifically so the hand-engineered blend is checked against the alternative rather than assumed superior.

Primary metrics are log-loss and Brier score. Accuracy, calibration curves, Map 1 versus Map 2+ strata, roster-continuity strata, and match-cluster bootstrap confidence intervals will remain secondary diagnostics. A new model advances only if it clears the predefined statistical guardrail on validation.

**Hyperparameter tuning (Lead directive — unchanged from Version 1.0, restated for Version 2.0):** all XGBoost hyperparameter search happens against the 10% Validation split only, minimizing validation log-loss, with the same bootstrap-CI guardrail used in Version 1.0 (a tuned configuration is only accepted if it clears the untuned baseline outside the confidence interval, guarding against the optimizer fitting Validation noise rather than a real improvement). Test is not touched during this process under any circumstance.

## Initial deliverables

- Data-source and licensing audit.
- Canonical player identity table and roster timeline.
- Point-in-time player feature builder with leakage tests.
- Player Elo update rule (centered `relative_performance_p` adjustment, `K_perf`) with `has_rich_stats` flagging.
- Time-decay implementation, both rating-level (team and player) and feature-level, each with a configurable half-life.
- Coverage and cold-start report.
- `HybridElo(w)` sweep notebook: `std(BaseElo_diff)` vs. `std(PlayerAggElo_diff)` scale check, the 101-point `w` grid, the bootstrap-CI guardrail against `w = 0.5`, and the Map 1 vs. Map 2+ stratified `w*` report — all completed and locked before the full XGBoost ablation.
- Ablation notebook and validation comparison against Version 1.0, following the six-step sequence above, including the Dynamic Hybrid Elo vs. two-feature comparison.
