# Local validation

Validated on 2026-11-07, Windows, Python 3.13.5.

- NumPy 2.3.2, pandas 3.0.5, scikit-learn 1.9.1, Matplotlib 3.10.3.
- Streamlit 1.67.0, pytest 9.2.0.
- Eight tests passed in 9.68 seconds.
- Tests cover deterministic/disjoint splits, baseline comparison, schema errors, missing/infinite/negative values, duplicate profiles, feature ordering, prediction submission and extrapolation warning.
- Training CLI ran successfully and exported metrics, coefficients, holdout predictions and diagnostic plots.
- The diagnostic plot was visually inspected.

Streamlit was checked with its official AppTest harness, not a manual browser session. Cloud deployment and hosted GitHub Actions have not been run. Scientific packages came from the existing environment; Streamlit and pytest were downloaded into a separate local dependency folder for validation. That folder is not distributed.
