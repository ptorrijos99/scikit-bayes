# Contributing to scikit-bayes

Welcome to `scikit-bayes`! We appreciate your interest in contributing to this project. We aim to build a robust, scalable library for Bayesian classifiers within the Python ecosystem.

## How to Contribute

1. **Bug Reports & Feature Requests:**
   Use the GitHub issue tracker to report bugs or suggest new features. Please include minimal reproducible examples for bugs.

2. **Pull Requests:**
   - Fork the repository.
   - Create a feature branch (`git checkout -b feature/my-new-feature`).
   - Implement your changes.
   - Ensure all tests pass. We strictly adhere to the `scikit-learn` estimator API.
   - Run the linting and formatting tools.
   - Submit a pull request.

## Development Environment

We use `pixi` for environment management, but you can also use `conda` or standard Python virtual environments.

### Setup (using Pixi)
```bash
git clone https://github.com/ptorrijos99/scikit-bayes.git
cd scikit-bayes
pixi install
```

### Running Tests
We use `pytest` for unit testing. Our goal is to maintain >95% test coverage.
```bash
# Run tests
pixi run test

# Alternatively, using native pytest
pytest skbn/
```

### Code Style (Linting & Formatting)
We strictly follow `black` for formatting and `ruff` for linting.
```bash
pixi run lint
```

## Adding New Estimators

If you are adding a new estimator:
- It must inherit from `sklearn.base.BaseEstimator`.
- It must pass `sklearn.utils.estimator_checks.check_estimator`.
- It must declare its parameters in `_parameter_constraints` and decorate `fit` with `sklearn.base._fit_context`, as the existing estimators do.
- It must have comprehensive docstrings in numpy style, an entry in `doc/api.rst` and a line in `CHANGELOG.md`.
- It must not mutate `self` state inside `predict` or `predict_proba` methods.
- Its tests must check values, not only that the code runs: against a reference implementation, a hand-computed case or an equivalent scikit-learn model.
- Mixed data should go through `MixedNB` (see `TAN` and `KDB`), so that the attribute types, the encoding of categories and the handling of unseen values stay the same across the library.

Every source file starts with the licence header:

```python
# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause
```

Thank you for helping us improve `scikit-bayes`!
