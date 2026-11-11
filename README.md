# scikit-bayes

[![tests](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/python-app.yml/badge.svg)](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/ptorrijos99/scikit-bayes/graph/badge.svg)](https://codecov.io/gh/ptorrijos99/scikit-bayes)
[![doc](https://github.com/ptorrijos99/scikit-bayes/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://ptorrijos99.github.io/scikit-bayes/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/1081042310.svg)](https://doi.org/10.5281/zenodo.23950884)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

scikit-bayes adds Bayesian network classifiers to scikit-learn. Every model follows the scikit-learn estimator API, so it works in pipelines, cross-validation and grid search, and takes tables that mix continuous, categorical and binary attributes, including pandas DataFrames with columns of strings or of `category` dtype. Counting and prediction run in compiled Numba kernels.

## Models

- `MixedNB`: Naive Bayes with Gaussian, categorical and Bernoulli attributes in one model.
- `AnDE`: Averaged n-Dependence Estimators (AODE, A2DE), pooled with the arithmetic or the geometric mean.
- `WeightedAnDE`: AnDE with discriminative weights, one per component or one per factor of each component (the weighted CCBN of Zaidi et al., 2017), at four levels of granularity, fitted jointly or one component at a time (which gives the AnDE<sup>w</sup> of FedAnDE).
- `AnJE` and `ALR`: Averaged n-Join Estimators and Accelerated Logistic Regression as defined by Zaidi et al. (2016), trained by default with the protocol of the paper.
- `GaussianAnJE` and `GaussianALR`: AnJE and ALR built on the factors of the SPnDEs, so that continuous attributes keep Gaussian conditionals, as in HAODE.
- `TAN`: Tree Augmented Naive Bayes, with a Chow-Liu tree of attribute dependencies.
- `KDB`: K-Dependence Bayesian classifier, in which every attribute has at most k attribute parents chosen by conditional mutual information.
- `WeightedTAN` and `WeightedKDB`: TAN and KDB with discriminative weights on their parameters (the weighted CCBN of Zaidi et al., 2017), which correct the dependencies the structure leaves out.
- `MDLPDiscretizer` and `DecisionTreeDiscretizer`: supervised discretization of continuous attributes, to place before a model in a pipeline. Every model also takes the MDL discretization of Fayyad and Irani directly, as `strategy="mdlp"`, and `continuous_children="discretized"` runs AnDE, TAN and KDB on the discretized attributes, as in the original works, instead of with Gaussian children.

## Quick start

```python
import numpy as np
from skbn import MixedNB, AnDE

# A Gaussian, a binary and a categorical column; the types are inferred
X = np.array([[1.5, 0, 2], [-0.5, 1, 0], [2.1, 1, 1], [-1.2, 0, 2]])
y = np.array([0, 1, 1, 0])

clf = MixedNB()
clf.fit(X, y)
print(clf.predict([[0.5, 1, 1]]))  # [1]

# XOR, which Naive Bayes cannot learn
X_xor = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

clf = AnDE(n_dependence=1, n_bins=2)
clf.fit(X_xor, y_xor)
print(clf.predict(X_xor))  # [0 1 1 0]
```

## Installation

```bash
pip install scikit-bayes
```

The development version installs from GitHub:

```bash
pip install git+https://github.com/ptorrijos99/scikit-bayes.git
```

The models run in Numba kernels that are compiled the first time each model is used and then cached on disk, so the first fit after installing takes from a couple of seconds (`MixedNB`) to some ten (`AnDE`), and later sessions start in a fraction of a second. To pay that once and check the installation:

```bash
python -m skbn
```

It prints the versions, fits and evaluates every model on the iris data and verifies the results.

## Documentation

- [User guide](https://ptorrijos99.github.io/scikit-bayes/user_guide.html): how each model is defined and estimated, and how to choose its parameters.
- [API reference](https://ptorrijos99.github.io/scikit-bayes/api.html): parameters, attributes and methods of every class.
- [Example gallery](https://ptorrijos99.github.io/scikit-bayes/auto_examples/index.html): what each model captures, shown on small datasets with plots.

## Development

The development environments are managed with [pixi](https://pixi.sh):

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
  author = {Torrijos, Pablo and Alcarria Laserna, {\'O}scar and G{\'a}mez, Jos{\'e} A. and Puerta, Jos{\'e} M.},
  title = {scikit-bayes: Bayesian Network Classifiers for Python},
  year = {2026},
  doi = {10.5281/zenodo.23950884},
  url = {https://github.com/ptorrijos99/scikit-bayes}
}
```

The DOI always resolves to the latest version; every release also has its own DOI on [Zenodo](https://doi.org/10.5281/zenodo.23950884) (1.3.0: [10.5281/zenodo.23950885](https://doi.org/10.5281/zenodo.23950885)). The same information is in [`CITATION.cff`](CITATION.cff), which GitHub shows as "Cite this repository".

## References

- Webb, G. I., Boughton, J., & Wang, Z. (2005). *Not so naive Bayes: Aggregating one-dependence estimators*. Machine Learning, 58(1), 5-24.
- Flores, M. J., Gámez, J. A., Martínez, A. M., & Puerta, J. M. (2009). *GAODE and HAODE: Two proposals based on AODE to deal with continuous variables*. ICML '09, 313-320.
- Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., & Cerquides, J. (2016). *ALR^n: Accelerated higher-order logistic regression*. Machine Learning, 104(2-3), 151-194.
- Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W., Hynes, M., & De Sterck, H. (2017). *Efficient parameter learning of Bayesian network classifiers*. Machine Learning, 106(9-10), 1289-1329.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.