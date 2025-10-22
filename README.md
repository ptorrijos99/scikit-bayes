# scikit-bayes

[![tests](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/python-app.yml/badge.svg)](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/ptorrijos99/scikit-bayes/graph/badge.svg)](https://codecov.io/gh/ptorrijos99/scikit-bayes)
[![doc](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://ptorrijos99.github.io/scikit-bayes/)

**scikit-bayes** is a Python package that extends `scikit-learn` with a suite of Bayesian Network Classifiers.

The primary goal of this package is to provide robust, `scikit-learn`-compatible implementations of advanced Bayesian classifiers that are not available in the core library. This includes models that relax the feature independence assumption of Naive Bayes and estimators that can natively handle mixed data types.

## Installation

Once available on PyPI, you can install the package using pip:

```bash
pip install scikit-bayes