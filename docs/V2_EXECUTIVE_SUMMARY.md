# Version 2.0: From Team Elo to Roster-Aware Forecasting — and a Real-World Test

Version 1.0 shipped on six columns: canonical team Elo, global and per-map, for both sides of a match. It worked, and it was deliberately minimal — every richer feature tried against it (static per-map rates, head-to-head history, rolling form) got tested and rejected by the same statistical discipline this document keeps returning to. But it had one structural blind spot that no amount of tuning could fix: **it had no idea who was actually on the server.**

## The problem

Team Elo tracks an organization's results. It does not track people. If a team makes a roster move — a stand-in for a banned player, a mid-season transfer, a permanent lineup change — Version 1.0's rating for that team doesn't move until the *new* lineup has played enough matches under the *same team identity* to shift it. In the interim, the model is confidently rating a team based on players who are no longer on it. For a competitive scene where rosters change constantly, that's not an edge case — it's a standing gap.

## The solution: Dynamic Hybrid Elo

Version 2.0 adds a second, independent rating layer — individual Player Elo, derived from team outcomes rather than invented from scratch, since CS2 box scores don't attribute a map win to one player. The two layers are then fed to the classifier separately (`BaseElo_diff`, a variance-matched `PlayerAggElo_diff`, and supporting context features), and it's the model — not a hand-picked formula — that learns how much weight each one deserves. That design decision was itself tested, not assumed: a hand-engineered single blended rating lost to the learned two-feature version on Validation, and the learned version is what shipped.

**Two decay rates, discovered independently, not assumed.** Team Elo and Player Elo don't erode at the same speed, and the project didn't guess at the difference — both half-lives were swept and guardrailed on Validation:

- **Team half-life: 180 days.** Tactical systems, coaching, roster chemistry — these visibly erode over a few months without play.
- **Player half-life: 1,095 days (roughly three years).** Within the realistic range of gaps a Tier-1 player actually produces — days to a couple of months between recorded appearances — individual skill barely needs forgetting at all. An earlier version of this exact sweep was corrupted by a datetime-unit bug that made every computed gap read near-zero regardless of true elapsed time; once caught and fixed, the signal was unambiguous. Individual mechanical skill and game sense behave as a durable, persistent trait. Team cohesion does not.

**Individual performance-adjustment was tested and rejected — `K_PERF = 0`.** The natural next idea was to scale each player's rating update by their box-score performance that specific map (more credit for a hard carry, less for an off night). Tested across 400 grid combinations, corroborated consistently across the *entire* tested half-life range: it added noise, not signal. A single map's ADR/KAST/K-D line is too confounded by round context — utility, executes, eco rounds, the specific opponent — to reliably isolate individual skill beyond what the team's own win or loss already carries. This is the fourth time in this project a hand-engineered box-score adjustment lost to simpler outcome-based Elo, after static per-map rates, head-to-head counts, and rolling form all failed the same way in Version 1.0. The lesson generalizes: in this domain, sophistication that isn't backed by genuinely new information doesn't survive contact with a real statistical test.

## The discipline

Every one of those decisions passed through the same gate: propose it, isolate it, and let a match-cluster bootstrap comparison on Validation decide — not a single point estimate. The clearest example is what *didn't* make the cut. A 75-trial Optuna hyperparameter search found a configuration that looked better on paper (lower Validation log-loss than the untuned baseline). The guardrail said otherwise: a 5,000-iteration paired bootstrap on the log-loss delta produced a 95% CI of **[-0.0042, +0.0005]** — straddling zero. Not a real improvement, statistically indistinguishable from noise. **The model that shipped uses the same fixed hyperparameters locked in Version 1.0, not the Optuna result.** Once the feature set correctly reflects the underlying reality — persistent individual skill, decaying team identity, a learned rather than hand-forced blend between them — there was no more headroom left for hyperparameter search to find. What remains is the real, irreducible variance of competitive Counter-Strike, not a modeling gap.

On the locked Test set (595 rows, including the full IEM Cologne 2026 holdout, evaluated exactly once): accuracy improved from 66.6% to 68.1%, log-loss from 0.6469 to 0.6099.

## The validation: a real, blind tournament backtest

Numbers on a held-out test set are one kind of evidence. A tournament that had already happened, forecast from a state snapshot frozen *before* it started, is another — closer to what this system actually needs to do. StarLadder StarSeries Fall 2026 (Barcelona) gave us that test: an 8-team double-elimination bracket, real verified rosters for all 40 players, model state locked at **2026-09-17T00:00:00**, one million Monte Carlo iterations per seed, and the real result — Vitality defeated Aurora 3-1 in the grand final — held out of every upstream computation until after the run was exported and locked.

**Champion call:**

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

Vitality was the clear favorite, by nearly 2x the next-closest team — and Vitality won.

**The Cinderella call.** Aurora's title odds (9.46%) put them just under the underdog threshold. But the model's estimate of Aurora reaching the *grand final anyway* — the "Cinderella" metric this project built specifically to answer "does an underdog have a real path, even without the title" — was **21.93%**. In reality, Aurora reached the grand final. This is exactly the kind of result the metric was designed to surface: not "who wins," but "who has a real path that a bare title-odds number would undersell."

**Bracket-path proximity, reported honestly.** The system also tracks the single most probable exact bracket path (every match winner, upper bracket through grand final) — but the top 5 paths by probability are statistically bunched (the leading path beats the runner-up by only 0.012 percentage points across nearly 12,000 distinct realized paths), so no single path should be read as a confident prediction. What is worth reporting: among all tracked paths, the highest-ranked one matching the real grand-final outcome (Vitality defeating Aurora) placed **11th**, at 0.1476% — just 0.026 percentage points behind the 5th-place cutoff. That's evidence the model assigns real, non-trivial density to the correct final matchup, not a claim that it nearly called the tournament's full round-by-round route — the real intermediate-round results aren't independently verified here, so that stronger claim isn't one this report can make.

**Stability, confirmed not assumed.** Two independent seeds, one million iterations each: the largest disagreement across every tracked metric (champion odds, grand-final matchups, runner-up odds, exact podiums, Cinderella reach) was 0.13 percentage points — well inside the 0.5-point tolerance set before the run. Every one of the 40 player identities and all 8 team identities resolved cleanly, with zero unintended cold starts, and the leakage boundary held exactly at the locked cutoff.

## Where this leaves the project

Not with a claim that more features always help — the record this cycle is mostly rejections, and that's the point. What moved the needle was structural correctness: giving team and individual identity their own, independently-validated decay dynamics, and being disciplined enough to let a learned blend beat a more elegant-looking hand-engineered one. The StarLadder backtest didn't just produce a good-looking number — it validated the two things this architecture was specifically built to catch: a clear favorite, correctly identified, and a real underdog path, correctly surfaced, in a tournament the system had never seen.
