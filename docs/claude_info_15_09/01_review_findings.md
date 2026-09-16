# Red Team Review Findings — 2026-09-15

Scope: `notebooks/1.5_locked_test_artifact_evaluation.ipynb`, `notebooks/1.6_monte_carlo_fast_path_and_cologne.ipynb`, `src/models/simulator.py`, `src/models/bracket_simulator.py`. Reviewed by reading source directly (code + notebook cells), not by re-running.

Severity scale: **Blocker** (must fix before the next live run) / **High** (fix soon, real risk) / **Medium** (latent risk, fix opportunistically) / **Info** (confirmed clean, no action).

---

## Finding 1 — Blocker: Locked Test metric and the deployed model are two different artifacts

**What's wrong:** the model that produced the officially locked Test number is not the model that will make live predictions.

- `notebooks/1.5_locked_test_artifact_evaluation.ipynb` originally trained `final_model` on **`X_train_val`** (Train+Val concatenated) and evaluated **raw, uncalibrated** `predict_proba` directly against Test. This finding was subsequently fixed in that notebook by loading the deployed calibrated artifact.
- `notebooks/1.6_monte_carlo_fast_path_and_cologne.ipynb` trains `canonical_model` on **Train only**, then fits isotonic calibration (`CalibratedClassifierCV(..., method='isotonic')`) on **Val only**. This calibrated artifact (`artifacts/map_classifier/canonical_elo_isotonic.joblib`) is what `SeriesSimulator` and `bracket_simulator.simulate_tournament` consume.

**Consequence:** the "64.6% Map1 accuracy, official and locked" figure describes a model that will never be used to predict a real match. The model that *will* be used live has never had its Test-set performance directly measured — nb06 only reports Val log-loss/Brier before/after calibration.

**Required action (pick one, don't leave both standing):**
1. Re-run the Test evaluation against the actual deployed artifact (`canonical_elo_isotonic.joblib`, Train-only + Val-calibrated), so the reported number describes the real system, **or**
2. Retrain that artifact on Train+Val with a proper held-out calibration slice carved out of it, so training data usage matches nb05 while still producing a calibrated model.

This should be resolved as a single, transparent, documented action (old number → new number → reason), consistent with the project's one-shot-Test discipline — this is a genuine correctness fix, not "peeking to improve a score."

**Related, unresolved:** the "9z"/"BETBOOM" join-key alias bug (missing `"team"` suffix, discovered defaulting these teams to 1500 Elo) was reported fixed, but its scope was never confirmed back to this reviewer. `notebooks/06...` cell 20 now uses `"9z team"` / `"betboom team"` (suffix present), suggesting the fix landed — but whether it touched the canonical Train/Val/Test generation pipeline (meaning nb05's number is also stale for this reason) or was a local patch was never confirmed. Resolve this alongside Finding 1, not separately — if the alias fix requires regenerating Train/Val/Test, do it as part of the same re-run.

---

## Finding 2 — Blocker: `bracket_simulator.py` hardcodes one specific single-elimination topology

**What's wrong:** the module encodes the exact IEM Cologne 2026 bracket shape as an implicit assumption, with no structural validation against the topology actually being simulated.

- `simulate_tournament` requires exactly 4 quarterfinal pairs (8 teams) — [`bracket_simulator.py:238-239`](../../src/models/bracket_simulator.py#L238-L239).
- Semifinal pairing is hardcoded by list position: `SF1 = winner(QF[0]) vs winner(QF[1])`, `SF2 = winner(QF[2]) vs winner(QF[3])` — [`bracket_simulator.py:338-355`](../../src/models/bracket_simulator.py#L338-L355). Nothing checks that the caller's `quarterfinals` ordering matches the real published bracket seeding.
- `_assert_tournament_probabilities` hardcodes stage totals for exactly this shape (`"QF": 8*n, "SF": 4*n, "Final": 2*n`) — [`bracket_simulator.py:612-617`](../../src/models/bracket_simulator.py#L612-L617).
- The whole module assumes **single elimination**: one loss ends a team's run. There is no concept of an upper/lower bracket, no logic for a team dropping to a loser's bracket after a single loss, no redemption-match handling.

**Consequence, now confirmed concrete:** StarLadder StarSeries Fall 2026 (the next live target, Sept 17) is a **double-elimination** bracket (see `03_tournaments_overview.md`). The current module cannot represent this topology at all — pointing it at StarLadder isn't a hardcoding tweak, it needs real double-elimination logic (loser-bracket progression, tracking which bracket a team is in, redemption match pairing) before it can run.

**Required action:**
1. Before any further hardcoding work: confirm whether the *next* live run is StarLadder (needs double-elim support built first) or can be deferred to PGL Bucharest / IEM Beijing (both confirmed single-elimination-after-Swiss, compatible with the current architecture — see overview doc).
2. If StarLadder is in scope, `bracket_simulator.py` needs a double-elimination code path added — this is new engineering, not a parameter change.
3. Regardless of format, add a validation step wherever a bracket is instantiated for a live event: the `quarterfinals`/seeding input must be checked against the actual published bracket graphic by a second pair of eyes before a live run — no assertion in this codebase can catch a transcription mistake in the seeding order, since it's a semantic error, not a mathematical one.

---

## Finding 3 — High: fast-inference path has no runtime guard on feature order

**What's wrong:** `FastIsotonicXGBoostPredictor.predict_numpy` calls `self.booster.inplace_predict(matrix)` on a bare NumPy array with no column names — [`bracket_simulator.py:80-95`](../../src/models/bracket_simulator.py#L80-L95). This is the source of the speed win (bypassing `CalibratedClassifierCV`'s per-call validation), but it means nothing checks that the hardcoded 6-column order (`N_ELO_FEATURES` / the tuple order used throughout `_simulate_series_batch`) still matches the booster's actual trained feature order.

**Why it matters:** this was validated once, empirically, against one specific model artifact (5.96×10⁻⁸ precision delta vs the slow path). If the model is ever retrained with a different feature-construction order — easy to do by accident when a notebook is rerun with a slightly reordered `FEATURE_COLUMNS` list — this path will silently produce wrong predictions. A raw Booster given a column-less array does not validate feature identity the way the sklearn wrapper does; there would be no error, just quietly wrong probabilities feeding the whole tournament simulation.

**Required action:** in `FastIsotonicXGBoostPredictor.from_calibrated_classifier`, assert `self.booster.feature_names == list(EXPECTED_ORDER)` at construction time (the booster stores `feature_names` when trained via a DataFrame with named columns). Fail loudly and immediately if it doesn't match, rather than relying on a one-time manual precision check that goes stale silently after any retrain.

---

## Smaller findings

- **`TEAM_DISPLAY_NAMES`** ([`bracket_simulator.py:19-28`](../../src/models/bracket_simulator.py#L19-L28)) is hardcoded to exactly the 8 Cologne team keys. Cosmetic only — falls back to the raw canonical join-key via `.get()` — but every team in a future tournament not in this dict will print as its lowercase join-key (e.g. `"navi"` instead of `"NAVI"`) unless extended or made data-driven before the next live run's summary table is generated.
- **`SeriesSimulator.__init__`** defaults `random_state: int | None = 42` at the class level ([`simulator.py:80`](../../src/models/simulator.py#L80)). Doesn't affect the bracket pipeline (which threads its own explicit `rng` and doesn't reuse this class), but is a latent footgun in the standalone series API: repeated instantiation for "independent" matchups without explicitly overriding the seed each time produces correlated random streams, not independent ones.
- **Random map-order sampling** (uniform, no veto realism) for any matchup without a known historical order is not a code bug — it's an accepted, previously-agreed limitation — but is flagged here because it is currently the single largest source of real uncertainty for any near-term live run: for a tournament starting in ~2 days, almost the entire bracket beyond round 1 is genuinely undetermined, so most of the simulated probabilities will rest on this simplification. Worth surfacing explicitly in any live-run write-up, not left as a buried footnote.

---

## Confirmed clean (no action needed)

- **No new data leakage found in the simulation layer.** Both snapshot-construction sections in `1.6_monte_carlo_fast_path_and_cologne.ipynb` correctly filter history with a strict `<` cutoff on the event's own start timestamp before building the Elo snapshot — no simulated match can see its own or a later result.
- **Calibration in nb06 is correctly fit on Validation only**, never Test.
- **The bootstrap CI in nb05** (cell 9) correctly resamples by `match_id`, not by row — properly respecting both the symmetrization duplication and multi-map series correlation. Good statistical practice, not a shortcut.
- **Diff-feature sign-flip assertions** are present and correctly wired in both notebooks' symmetrization functions.
