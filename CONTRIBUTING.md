# Contributing

Thank you for improving the CS2 forecasting project. Contributions are welcome when they preserve the temporal and statistical guarantees that make the reported results meaningful.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Use a short-lived branch from `main`. Keep changes focused and include the evidence needed to review them.

## Before opening a pull request

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall .
python scripts/check_markdown_links.py
git diff --check
```

The pull request should explain:

- what changed and why;
- which temporal boundary and dataset were used;
- whether any model artifact or published metric changed;
- what tests or statistical comparisons support the change;
- whether generated files were intentionally updated.

## Modeling rules

- Features must be available strictly before the predicted map begins.
- Stateful features must be emitted before applying the current result.
- Fit preprocessing on Train and calibration on Validation only.
- Do not inspect Test to select features, parameters, thresholds, or narratives.
- Use canonical team and player identities and report cold starts explicitly.
- Compare candidate improvements with paired match-cluster uncertainty, not point estimates alone.
- Preserve the exact stored feature order at inference.

Any proposal that changes these rules needs an explicit design note and new leakage tests.

## Data and artifacts

- Do not commit secrets, credentials, private data, or unlicensed source material.
- Do not overwrite timestamped production results.
- New model artifacts require metadata with their SHA-256, feature contract, training/calibration periods, seed, and software assumptions.
- Generated parquet and Joblib files are binary; review their accompanying metadata rather than relying on a binary diff.
- Keep heavy Monte Carlo runs out of CI.

## Code and tests

- Prefer small, testable classes and pure simulation primitives.
- Preserve deterministic ordering by timestamp and stable tie-break keys.
- Add a regression test for every leakage, identity, state-mutation, or bracket-invariant bug.
- Keep the fast inference path numerically aligned with its reference implementation.

## Documentation

- English is the primary public language.
- Educational notebooks preserve the English-first, Hebrew-second format.
- Distinguish current specifications from planning or historical documents.
- Update `docs/RESULTS.md` only from locked artifact evidence.
- Keep repository-relative links valid by running `scripts/check_markdown_links.py`.

## Scope and conduct

Keep discussion technical, evidence-based, and respectful. This project is for research and education; contributions should not present model probabilities as guaranteed outcomes or betting advice.
