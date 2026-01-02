# scikit-bayes

[![tests](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/python-app.yml/badge.svg)](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/ptorrijos99/scikit-bayes/graph/badge.svg)](https://codecov.io/gh/ptorrijos99/scikit-bayes)
[![doc](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://ptorrijos99.github.io/scikit-bayes/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**scikit-bayes** is a Python package that extends `scikit-learn` with a suite of Bayesian Network Classifiers.

The primary goal of this package is to provide robust, `scikit-learn`-compatible implementations of advanced Bayesian classifiers that are not available in the core library.

## Key Features

- **MixedNB**: Naive Bayes for mixed data types (Gaussian + Categorical + Bernoulli) in a single model
- **AnDE**: Averaged n-Dependence Estimators (AODE, A2DE) that relax the independence assumption
- **ALR**: Accelerated Logistic Regression - hybrid generative-discriminative models with 4 weight granularity levels
- **WeightedAnDE**: Discriminatively-weighted ensemble models
- **Full scikit-learn API**: Compatible with pipelines, cross-validation, and grid search

## Quick Start

```python
import numpy as np
from skbayes import MixedNB, AnDE

# MixedNB: Handle mixed data types automatically
X = np.array([[1.5, 0, 2], [-0.5, 1, 0], [2.1, 1, 1], [-1.2, 0, 2]])
y = np.array([0, 1, 1, 0])

clf = MixedNB()
clf.fit(X, y)
print(clf.predict([[0.5, 1, 1]]))  # Automatically handles Gaussian, Bernoulli, Categorical

# AnDE: Solve problems Naive Bayes cannot (XOR)
X_xor = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

clf = AnDE(n_dependence=1, n_bins=2)
clf.fit(X_xor, y_xor)
print(clf.predict(X_xor))  # [0, 1, 1, 0] ✓
```

## Installation

```bash
pip install scikit-bayes
```

Or install from source:

```bash
pip install git+https://github.com/ptorrijos99/scikit-bayes.git
```

## Documentation

- 📖 [User Guide](https://ptorrijos99.github.io/scikit-bayes/user_guide.html) - Detailed documentation
- 📚 [API Reference](https://ptorrijos99.github.io/scikit-bayes/api.html) - Complete API docs
- 🎨 [Examples Gallery](https://ptorrijos99.github.io/scikit-bayes/auto_examples/index.html) - Visual examples

## Development

This project uses [pixi](https://pixi.sh) for environment management.

```bash
# Run tests
pixi run test

# Run linter
pixi run lint

# Build documentation
pixi run build-doc

# Activate development environment
pixi shell -e dev
```

## Citation

If you use scikit-bayes in a scientific publication, please cite:

```bibtex
@software{scikit_bayes,
  author = {Torrijos, Pablo},
  title = {scikit-bayes: Bayesian Network Classifiers for Python},
  year = {2025},
  url = {https://github.com/ptorrijos99/scikit-bayes}
}
```

## References

- Webb, G. I., Boughton, J., & Wang, Z. (2005). *Not so naive Bayes: Aggregating one-dependence estimators*. Machine Learning, 58(1), 5-24.
- Flores, M. J., Gámez, J. A., Martínez, A. M., & Puerta, J. M. (2009). *GAODE and HAODE: Two proposals based on AODE to deal with continuous variables*. ICML '09, 313-320.
- Zaidi, N. A., Webb, G. I., Carman, M. J., & Petitjean, F. (2017). *Efficient parameter learning of Bayesian network classifiers*. Machine Learning, 106(9-10), 1289-1329.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.