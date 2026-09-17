"""Insert an English companion immediately before every Hebrew Markdown cell."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
HEBREW = re.compile(r"[\u0590-\u05ff]")


ENGLISH_MARKDOWN = {
    "1.0_setup_and_data_sanity.ipynb": [
        "# CS2 Tournament Simulator: Setup and Data Validation\n\nThis notebook validates the project environment, loads the initial feature table, and applies an explicit column allowlist before modeling begins.",
        "## Environment Verification\n\nAll core libraries are imported and their versions are recorded for reproducibility. An import failure stops the workflow before partial or misleading outputs can be created.",
        "## Loading the Source Table\n\nThe path works from either the project root or `notebooks/`. The full table is loaded first so required columns can be validated explicitly before filtering.",
        "## Feature Allowlist and Leakage Prevention\n\nOnly identifiers, time, map, target, and the original DNA candidates are retained. Post-match statistics are excluded; later ablations ultimately restrict production to six canonical Elo features.",
    ],
    "1.1_point_in_time_elo_pipeline.ipynb": [
        "# Point-in-Time Elo Pipeline — Version 2\n\nThis notebook builds a probability baseline using historical map results only, measuring the predictive value available before each map starts.",
        "## Hierarchical Elo Design\n\nEach team has a global rating and a separate rating for every map. Features are captured before the current result updates either rating.",
        "## Chronological Loading, Filtering, and Construction\n\nThe engine removes invalid map and series-total rows, orders observations chronologically, emits pre-match ratings, and only then applies the observed result.",
        "## Locked Time Split\n\nTraining ends on 2025-12-31, validation covers 2026 Q1, and test begins in 2026 Q2. No future period contributes information to an earlier split.",
        "## Raw Elo Baseline on Validation\n\nAccuracy, log-loss, and Brier score evaluate both decisions and probability quality. The raw global-Elo probability establishes the minimum benchmark for learned models.",
        "## Part 2 Summary and Transition to Ablation\n\nThe raw baseline is now fixed. The next experiment tests whether XGBoost can calibrate Elo without mixing in unrelated feature families.",
        "# Part 3 — Ablation Study: XGBoost Using Elo Only\n\nThis controlled experiment isolates Elo features so any change can be attributed to nonlinear calibration rather than additional information sources.",
        "## Symmetrization After the Time Split\n\nEach map receives a mirrored copy with teams exchanged and the target inverted. Splits are symmetrized independently to preserve chronological isolation.",
        "## XGBoost Training and Log-Loss Monitoring\n\nThe model uses only absolute Elo ratings, reliability counters, differences, and the raw prior. Validation log-loss is monitored during boosting.",
        "## Evaluation on Symmetrized Validation\n\nAccuracy, log-loss, Brier score, and a reliability diagram assess discrimination and calibration on the same independently transformed validation period.",
        "## Ablation Decision\n\nThe Elo-only result is recorded as the comparison point for all later candidate features and for the eventual canonical model selection.",
    ],
    "1.2_xgboost_elo_h2h_dna_candidate.ipynb": [
        "# Version 2 Candidate: Elo + H2H + DNA\n\nThis notebook documents a candidate that combined three feature families but was not selected for production.",
        "## Point-in-Time Elo and Head-to-Head History\n\nElo and prior matchup counts are emitted before the current map is processed, ensuring that every value was available at prediction time.",
        "## Joining DNA by Team and Map\n\nPistol, CT-side, and T-side rates are joined with observation counts and missingness flags. Sparse coverage remains explicit rather than being hidden by neutral imputation.",
        "## Locked Split and Independent Symmetrization\n\nTrain, validation, and test use the fixed calendar boundaries. Train and validation are mirrored separately only after the split.",
        "## Bayesian Optimization with Optuna\n\nA seeded TPE sampler explores tree depth, learning rate, sampling, and regularization while minimizing validation log-loss.",
        "## Refit of the Selected Configuration\n\nThe best parameters are refit on training data only, with validation used for early stopping and final candidate evaluation.",
        "## Reliability Diagram\n\nValidation predictions are grouped into probability bins to compare predicted confidence with observed win frequency.",
        "## Conclusion and Stop Decision\n\nThis richer candidate underperformed the Elo-only benchmark. The result motivates controlled ablations instead of assuming that more features improve the model.",
    ],
    "1.3_canonical_elo_feature_ablation.ipynb": [
        "# Canonical Elo Diagnostics by Map Position\n\nThis notebook isolates honest team-level Elo after canonical identity resolution and tests performance on Map 1 versus later maps.",
        "## Point-in-Time Features and Map Position\n\nRatings are captured before each update, while map position is derived within each series to expose possible intra-series momentum effects.",
        "## Symmetrization and Elo-Only Matrix\n\nTrain and validation are mirrored independently. Only global Elo, map Elo, and their differences enter the model.",
        "## Fixed Training Without Optuna\n\nA fixed conservative configuration measures feature quality rather than search-budget effects, with early stopping used only for iteration selection.",
        "## Overall and Map-Stratified Metrics\n\nAccuracy and Brier score are reported overall and separately for Map 1 and Map 2+, distinguishing pre-series forecasting from within-series prediction.",
        "## Step 1 Conclusion and Uncertainty Check\n\nThe observed map-position gap is descriptive until cluster bootstrap intervals determine whether it is distinguishable from sampling noise.",
        "# Step 1b — Match-Cluster Bootstrap Confidence Intervals\n\nEntire matches, not individual rows, are resampled so correlated maps and mirrored copies remain together.",
        "# Step 2 — Adding DNA to Canonical Elo\n\nPistol, CT-side, and T-side rates and their differences are added to test whether sparse round-level summaries improve canonical Elo.",
        "## Elo + DNA Symmetrization and Matrix\n\nTeam-specific DNA columns are exchanged in mirrored rows and differences are recomputed after the exchange.",
        "## Fixed Training and Map-Stratified Evaluation\n\nThe same XGBoost configuration is retained so metric changes reflect DNA rather than hyperparameter tuning.",
        "## Step 2 Conclusion and Transition to H2H\n\nDNA degrades the Elo signal, consistent with sparse noisy features attracting unhelpful tree splits. It is removed from the next experiment.",
        "# Step 3 — Elo + Head-to-Head Isolation\n\nStrictly prior matchup win counts are added without DNA to test their incremental value over canonical Elo.",
        "## Symmetrizing Head-to-Head Counters\n\nTeam counters are exchanged under mirroring, the difference changes sign, and the coverage indicator remains unchanged.",
        "## Fixed Training and Three-Way Evaluation\n\nMetrics are reported overall, by map position, and by whether previous head-to-head history exists.",
        "## Step 3 Conclusion and Transition to Rolling Form\n\nThe apparent benefit in covered matchups is attributed to selection bias: teams with H2H history also tend to have mature Elo ratings.",
        "# Step 4 — Elo + Five-Map Rolling Form\n\nPre-match win rate and round differential over each team's last five maps are tested as strictly chronological form indicators.",
        "## Rolling-Form Symmetrization and Matrix\n\nBoth teams' rolling features are exchanged in mirrored rows and their signed differences are recomputed.",
        "## Fixed Training and Map-Position Evaluation\n\nThe unchanged model configuration isolates the contribution of rolling form across overall, Map 1, and Map 2+ strata.",
        "## Final Ablation Decision\n\nRolling form also fails to improve the canonical Elo benchmark. DNA, H2H, and form are therefore excluded from the final production feature set.",
    ],
    "1.4_optuna_statistical_guardrail.ipynb": [
        "# Final V2 Tuning on Canonical Elo Only\n\nAfter the ablations, tuning is restricted to the six canonical Elo features so discarded noise cannot re-enter the model.",
        "## Point-in-Time Elo and Locked Time Split\n\nCanonical identities and deterministic chronological ordering produce pre-match features before the fixed train and validation boundaries are applied.",
        "## Symmetrization and Minimal Allowlist\n\nTeams are mirrored only after splitting, and the matrix contains four absolute ratings plus global and map Elo differences.",
        "## Reproducing the Fixed Baseline\n\nThe untuned Elo-only model is refit first so the tuned candidate is compared against a locally reproduced baseline.",
        "## TPE Optimization for Validation Log-Loss\n\nFifty seeded Optuna trials search depth, learning rate, sampling, and regularization, with validation log-loss as the objective.",
        "## Refit and Validation Stratification\n\nThe selected configuration is refit on training data and evaluated overall and by map position.",
        "## Paired Cluster-Bootstrap Guardrail\n\nBoth models predict the same rows, so matches are resampled jointly and paired metric differences preserve within-match dependence.",
        "## Tuning Decision\n\nBecause the tuned model does not statistically clear the fixed baseline, production retains the simpler default canonical-Elo configuration.",
    ],
    "1.5_locked_test_artifact_evaluation.ipynb": [
        "# Final One-Time Evaluation on the Locked Test Set\n\nThis notebook opens 2026 Q2 only after every modeling decision is frozen and evaluates the model intended for deployment.",
        "## Point-in-Time Feature Construction\n\nThe engine processes the full history in deterministic chronological order and emits ratings before applying each current result.",
        "## Identical Symmetrization Across Splits\n\nTrain, validation, and test are mirrored independently. Ratings swap, targets invert, and signed differences are recomputed.",
        "## Provenance Correction: Evaluate the Deployed Artifact\n\nThe deployed isotonic-calibrated artifact—not a newly trained raw model—is loaded and checked against the locked six-feature order.",
        "## Opening the Vault: Final Test Metrics\n\nTest predictions are used once to report log-loss, accuracy, and Brier score overall and by map position.",
        "## Final Conclusion\n\nThese are the official out-of-time metrics for the exact calibrated artifact used by the simulators and should not drive further model selection.",
    ],
    "1.6_monte_carlo_fast_path_and_cologne.ipynb": [
        "# Series Simulation and Backtest: Cologne 2026 Final\n\nThis notebook connects the calibrated canonical-Elo map classifier to a map-by-map Monte Carlo series simulator.",
        "## Environment Setup\n\nThe random seed and six-feature order are locked for reproducibility and artifact compatibility.",
        "## Point-in-Time Elo and Chronological Splits\n\nThe Elo engine reconstructs information available before each map and applies the established train, validation, and test windows.",
        "## Leakage-Free Symmetry\n\nTraining and validation are mirrored independently after splitting, with ratings exchanged and differences recalculated.",
        "## Locked Model Training and Isotonic Calibration\n\nThe base classifier is trained on Train and calibrated strictly on Validation using isotonic regression.",
        "## Persisting and Reloading the Inference Artifact\n\nThe calibrated model and feature order are serialized together, reloaded immediately, and verified before simulation.",
        "## Freezing Elo Before the Final\n\nOnly maps completed before the Cologne final are processed, producing the immutable starting state for both finalists.",
        "## Monte Carlo with In-Series Momentum\n\nEach of 5,000 series begins from the frozen state, predicts each map, then updates an in-memory Elo copy before the next map.",
        "## Reading the Result\n\nFalcons' win probability is the sum of its 3–0, 3–1, and 3–2 scoreline probabilities; the remaining outcomes belong to FURIA.",
        "# Full Playoff-Bracket Simulation\n\nThe same series primitive is expanded to an eight-team knockout bracket with Elo state threaded through every advancing team.",
        "## Bracket Topology and Map Orders\n\nHistorical quarterfinal pairings and known vetoes are used where available; otherwise maps are sampled without replacement from the active pool.",
        "## Pre-Playoff Elo Snapshot\n\nA new snapshot includes only results completed before the first playoff quarterfinal and is shared as the starting point for every path.",
        "## Pure Series Function and State Threading\n\n`simulate_series_once` copies its input state, applies live updates locally, and returns both the winner and updated state without global mutation.",
        "## 10,000 Tournaments and Sanity Checks\n\nThe progress-enabled Monte Carlo counts stage appearances and asserts normalized champion probabilities and monotonic advancement probabilities.",
        "## Cautious Interpretation\n\nThe table is a pre-playoff forecast, not a retrospective ranking; all teams start in the quarterfinal stage by construction.",
        "# Canonical Identity Audit\n\nThe audit confirms the historical keys `9z team` and `betboom team` are consistent across feature generation and simulation lookup.",
        "# Phase 2: Stability at 100,000 Tournaments\n\nThe bracket is simulated with seeds 42 and 99 to verify that champion estimates are stable at production scale.",
        "## High-Resolution Summary\n\nSeed 42 serves as the primary table while seed 99 provides an independent Monte Carlo stability check.",
    ],
    "1.7_double_elimination_preproduction.ipynb": [
        "# StarLadder Operation: Live Double-Elimination Simulation\n\nThis notebook documents the controlled pre-production forecast for StarLadder StarSeries Fall 2026 using the deployed calibrated artifact.",
        "## Locked Tournament Configuration\n\nThe four official opening matchups are ordered exactly as seeded because their positions determine all later upper- and lower-bracket paths.",
        "## Point-in-Time Elo Snapshot and Artifact Verification\n\nOnly maps completed before tournament start build the snapshot, while feature order and the model SHA-256 are verified before inference.",
        "## Two Full-Scale Monte Carlo Runs\n\nEach seed simulates 100,000 tournaments with a consistent progress display and independent in-tournament Elo state for every path.",
        "## Empirical Stability Gate\n\nThe run is accepted only when the maximum team-level difference in champion probability between seeds is at most 0.5 percentage points.",
        "## Atomic Export and Provenance Chain\n\nCSV, JSON, and metadata files are written through temporary files, indexed globally, and tied to the exact model artifact by SHA-256.",
        "## Operational Interpretation\n\nOutputs are pre-tournament forecasts based on historical Elo, isotonic calibration, and uniform unknown-map sampling—not knowledge of future vetoes or results.",
    ],
    "1.8_starladder_1m_production.ipynb": [
        "# Production Run: One Million Simulations and Rare-Event Analytics\n\nThis notebook launches the approved StarLadder production run for two independent seeds and measures stable low-frequency outcomes.",
        "## Pre-Flight Equivalence Test\n\nBefore the million-path run, 5,000 paths are evaluated through both the NumPy/Booster fast path and `CalibratedClassifierCV`; all counters must match exactly.",
        "## Advanced Analytics\n\nEach path records Grand Final matchup, runner-up, exact podium, and Cinderella Final appearances in lightweight counters without retaining row-level traces.",
        "## Stability Gate and Atomic Export\n\nSeeds 42 and 99 are compared across every analytical category, then CSV, JSON, metadata, model hash, timing, and run-index records are written atomically.",
    ],
}


def english_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"language": "en", "paired_translation": True},
        "source": [line + "\n" for line in source.splitlines()][:-1]
        + ([source.splitlines()[-1]] if source.splitlines() else []),
    }


def main() -> None:
    for filename, translations in ENGLISH_MARKDOWN.items():
        path = NOTEBOOK_DIR / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        hebrew_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and HEBREW.search("".join(cell.get("source", [])))
        ]
        if len(hebrew_cells) != len(translations):
            raise AssertionError(
                f"{filename}: {len(hebrew_cells)} Hebrew cells but "
                f"{len(translations)} translations"
            )

        rebuilt = []
        translation_index = 0
        for cell in notebook["cells"]:
            text = "".join(cell.get("source", []))
            if cell["cell_type"] == "markdown" and HEBREW.search(text):
                if rebuilt and rebuilt[-1].get("metadata", {}).get(
                    "paired_translation"
                ):
                    rebuilt.pop()
                rebuilt.append(english_cell(translations[translation_index]))
                translation_index += 1
            rebuilt.append(cell)
        notebook["cells"] = rebuilt
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"{filename}: added {translation_index} English cells")


if __name__ == "__main__":
    main()
