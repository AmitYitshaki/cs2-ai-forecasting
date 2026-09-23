# Documentation Index

This directory contains both the current Version 2.0 specification and the historical records that explain how the project reached it. Use the reading order below to avoid mistaking an early hypothesis for a production decision.

## Recommended reading path

1. [Project README](../README.md) — problem, headline results, setup, and limitations.
2. [V2 Executive Summary](V2_EXECUTIVE_SUMMARY.md) — the short narrative: why roster-aware modeling mattered and what the backtest demonstrated.
3. [Current Architecture](ARCHITECTURE.md) — the deployed data, model, calibration, and simulation pipeline.
4. [Project Results](RESULTS.md) — canonical metrics, confidence intervals, and Monte Carlo tables.
5. [Reproducibility Guide](REPRODUCIBILITY.md) — safe commands for tests, rebuilds, and production simulation.

## Source-of-truth hierarchy

When two documents appear to disagree, use this order:

1. Artifact metadata under `artifacts/map_classifier/` for model inputs, hashes, and calibration.
2. Timestamped metadata under `results/starladder/` for simulation outputs.
3. [RESULTS.md](RESULTS.md) for the human-readable canonical report.
4. [ARCHITECTURE.md](ARCHITECTURE.md) for the current implementation.
5. Planning and historical documents for context only.

## Current V2 documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Concise technical description of the deployed V2 system |
| [RESULTS.md](RESULTS.md) | Locked Test metrics and final StarLadder Barcelona results |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, verification tiers, and artifact provenance |
| [V2_EXECUTIVE_SUMMARY.md](V2_EXECUTIVE_SUMMARY.md) | Portfolio-friendly project narrative |
| [V2_POST_MORTEM.md](V2_POST_MORTEM.md) | What survived the statistical guardrails and what did not |

## Design and historical records

| Document | Status | Why it remains useful |
|---|---|---|
| [V2_ARCHITECTURE_PLAN.md](V2_ARCHITECTURE_PLAN.md) | Design history | Records the hypotheses, formulas, and safeguards considered before the final ablations |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | V1 historical context | Documents the original datasets, quality findings, and project framing |
| [WORK_PLAN.md](WORK_PLAN.md) | V1 execution log | Preserves the staged build plan and completed checkpoints |
| [Kaggle_AI_Agents_Playbook.md](Kaggle_AI_Agents_Playbook.md) | Process reference | Hebrew engineering and educational-notebook conventions |
| [claude_info_15_09/](claude_info_15_09/) | Archived red-team package | Captures the V1 review that led to the double-elimination and artifact-provenance work |

Historical documents are retained deliberately. They show failed hypotheses, corrections, and review decisions instead of presenting the final system as if it appeared fully formed.

## Notebook sequence

| Range | Story |
|---|---|
| `1.0`–`1.5` | V1 data checks, point-in-time Elo, ablations, statistical guardrails, and locked Test evaluation |
| `1.6`–`1.8` | Fast Monte Carlo inference, double elimination, and the V1 production simulation |
| `2.0` | V2 player/team feature-store construction |
| `3.0` | Dynamic Hybrid Elo validation, XGBoost comparison, calibration, and one-shot Test evaluation |

Notebook explanations are bilingual: English first for international readers, followed by the original Hebrew teaching text.

## Maintenance rules

- Never edit a timestamped production result to make a narrative cleaner; generate a new artifact instead.
- Keep headline numbers synchronized with artifact metadata.
- Mark planning documents as historical once implementation decisions supersede them.
- Use repository-relative links and run `python scripts/check_markdown_links.py` before publishing.
- Do not describe Validation improvements as proven unless the predefined confidence-interval guardrail clears.
