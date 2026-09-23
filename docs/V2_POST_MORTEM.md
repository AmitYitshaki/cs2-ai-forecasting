# Version 2.0 Post-Mortem: What Actually Survived Contact With the Guardrails

Every hypothesis in this cycle got the same treatment: propose it, engineer it properly, isolate it, and let a bootstrapped Validation comparison decide whether it lives in the production feature set — not intuition, not a single point estimate. Most of what we tried didn't survive. What did survive moved Test log-loss from **0.6469 to 0.6099** and Test accuracy from **66.6% to 68.1%**. This is the record of both.

## The premise we started from

Version 1.0 shipped on six columns: global and per-map canonical Elo for two teams, and their diffs. It was deliberately minimal — every richer feature we tried (static DNA rates, head-to-head history, five-match rolling form) got tested and rejected by the same guardrail this document keeps referencing. The V1 postmortem's implicit thesis was: *team-identity Elo is close to the ceiling of what this dataset can support.*

Version 2.0 set out to test that thesis directly, on two fronts: **time decay** (should a rating that hasn't been touched in months still be trusted at face value?) and **individual player tracking** (does a team's Elo lag reality the moment a roster changes?).

## What we built

**A stateful Player Elo, derived from team outcomes, not invented from scratch.** CS2 box scores don't attribute a map win to one player — so each map's team-level Elo delta is redistributed across the five active players via a *centered* performance adjustment: `player_delta = team_delta/5 + K_perf × relative_performance`, where `relative_performance` is each player's impact centered against their own team's average that map. Centering guarantees the five deltas sum exactly to the team's delta regardless of win or loss — and, not incidentally, it's a correction to an earlier share-multiplicative version of this same formula that got the sign wrong on losses (a hard carry lost *more* points in a defeat, not fewer, until this was caught and fixed before it ever reached the data).

**Independent, empirically-fit decay for two different kinds of forgetting.** Team Elo and Player Elo don't erode at the same rate, and we didn't assume they did — both half-lives were swept on Validation, not hardcoded:

```
R_effective(t) = μ + (R_stored - μ) × 2^(-gap_days / half_life)
```

**Two ways to combine macro (team) and micro (player) signal, tested against each other, not chosen by preference.** Approach A hand-engineered a single blended `HybridElo` before the model ever saw it (`w · BaseElo + (1-w) · PlayerAggElo`, weight swept and guardrailed). Approach B fed `BaseElo_diff` and a variance-matched `PlayerAggElo_diff` to XGBoost as separate features and let the trees learn the interaction. We didn't assume the elegant hand-engineered version would win. **It didn't.** Approach B beat Approach A on Validation, and Approach B is what shipped.

## Four guardrails, four rejections, one acceptance

This is the part worth being honest about, because the interesting result isn't a list of features that worked — it's how much didn't, and why that's the finding.

| Hypothesis | Test | Result |
|---|---|---|
| Individual performance adjustment (`K_perf > 0`) helps Player Elo | Grid search, 400 combinations, corroborated across the *entire* tested half-life range | **Rejected.** `K_perf = 0` wins consistently. A single map's ADR/KAST/K-D line is too confounded by round context — utility, executes, eco rounds, that specific opponent — to add signal beyond what the team's own win/loss already carries. This is the fourth time in this project a hand-engineered box-score-derived adjustment has lost to simpler outcome-based Elo (after static DNA rates, head-to-head counts, and rolling form in V1). |
| Hand-blended `HybridElo` beats a learned blend | Fixed-hyperparameter comparison, both approaches | **Rejected.** The learned blend (Approach B) won. Elegance lost to letting the model find the interaction. |
| 75-trial Optuna search beats the untuned baseline | Match-cluster bootstrap, 5,000 iterations, paired per-row log-loss delta | **Rejected.** 95% CI on the delta: `[-0.0042, +0.0005]` — straddles zero. The untuned configuration (the same fixed hyperparameters locked in Version 1.0) is what's actually deployed. Once the feature set reflects the right signal, there was no more headroom left for hyperparameter search to find. |
| Disabling Player-Elo decay entirely beats a 1,095-day half-life | Point comparison only — not bootstrapped, and flagged here as exactly that | **Inconclusive, not "confirmed."** `0.6578` (no decay) vs. `0.6570` (1,095 days) is a gap of 0.0008 on 633 Validation rows with no confidence interval computed. Decay did not clearly help beyond a very long half-life — treat that as the honest claim, not a proven "physiological baseline." |

And the one hypothesis that *did* survive, and reshaped the whole model: **team identity and individual skill decay at genuinely different rates.** The guardrailed sweep landed on `TEAM_HALF_LIFE_DAYS = 180`, `PLAYER_HALF_LIFE_DAYS = 1095`. Team-level Elo — tactics, system, coaching, roster chemistry — visibly erodes over a matter of months. Individual skill, once measured correctly (an earlier version of this exact sweep was corrupted by a datetime-unit bug that made every computed gap read near-zero regardless of true elapsed time — caught and fixed before being trusted), behaves as a far more persistent trait. Within the realistic range of gaps a Tier-1 CS2 player actually produces — days to a couple of months — individual skill barely needs forgetting at all. That's not a coincidence of the search grid; it's a real, structurally sensible distinction the guardrails found on their own.

## The numbers

The deployed model: `BaseElo_diff` + variance-matched `PlayerAggElo_diff` + a handful of auxiliary context features (gap-days, roster dispersion, lineup continuity, cold-start counts), fed to XGBoost with the same fixed hyperparameters locked in Version 1.0. Evaluated exactly once against the 595-row Test set, including the full 187-row IEM Cologne 2026 holdout, never touched before this single evaluation.

| Version | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| V1 (canonical Elo only) | 66.6% | 0.6469 | 0.2142 |
| **V2 (this model)** | **68.1%** | **0.6099** | **0.2100** |

| V2 Test stratum | Rows | Accuracy (95% CI) | Log-loss (95% CI) |
|---|---:|---:|---:|
| Map 1 (cold start) | 294 | 61.6% (55.8–67.3%) | 0.6333 (0.608–0.658) |
| Map 2+ (in-series) | 301 | 74.4% (68.0–80.6%) | 0.5871 (0.557–0.618) |

The Map 1 / Map 2+ gap is the same structural story this project has told since Version 1.0's momentum discovery: once a series is underway, the model has far more to work with than a cold open. Map 1 remains the harder, more honest test of "do we actually understand these two teams" — and it's also where V2's gains matter most, since that's the case every live tournament forecast leans on hardest.

**IEM Cologne 2026** (187 rows, held out from every decision made this cycle): 69.0% accuracy, 0.6081 log-loss — consistent with the aggregate Test figures, no sign of overfitting to the rest of the holdout. The isolated Falcons–FURIA rows show a mean map-win probability of 65.1% for Falcons — directionally consistent with, but not directly comparable to, the original series-level Monte Carlo backtest's 70.4% (a series probability compounding multiple maps is a different estimand from a single map's mean probability; both point the same direction).

## What this cycle actually proved

Not that more features help. Four different attempts to inject richer signal — individual performance adjustment, a hand-engineered blend, aggressive hyperparameter search, and (inconclusively) aggressive decay — were tested and three were rejected outright, with the fourth left honestly unresolved rather than oversold. What moved the needle was **structural correctness**: giving team and individual identity their own, separately-validated decay dynamics, and being disciplined enough to let a learned blend beat our own hand-engineered formula rather than ship the more elegant-looking option anyway. The guardrails did their job. That's the actual headline.
