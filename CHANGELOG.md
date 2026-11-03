# Changelog

All notable changes to scikit-bayes are listed here. The project follows [semantic versioning](https://semver.org/).

## Unreleased (1.3.0)

### New estimators

- `TAN`: Tree Augmented Naive Bayes, with a Chow-Liu tree on the conditional mutual information (contributed by Óscar Alcarria Laserna, #15).
- `KDB`: K-Dependence Bayesian classifier of Sahami (1996) (contributed by Óscar Alcarria Laserna, #16).
- `GaussianAnJE` and `GaussianALR`: AnJE and ALR built on the factors of the SPnDEs, so that continuous attributes keep Gaussian conditionals.
- `AnJE` and `ALR` now follow the definitions and training protocol of Zaidi et al. (2016); `WeightedAnDE` gains weights per factor (`weight_scope="factor"`).

### Changed behaviour

- Categorical attributes are label-encoded on the values seen in training, so their codes may be negative, sparse or non-integer. A categorical value never seen in training now contributes nothing to the prediction (the attribute is marginalised), instead of being read as the closest category of the training range.
- Bernoulli attributes are binarised at 0 only when their training values lie on both sides of 0, and at the midpoint of their range otherwise. Two-valued attributes such as {1, 2} or {0.3, 0.7} were binarised as constant before and lost all their information.
- Undeclared integer-valued attributes with more than 20 distinct values (ages, counts, amounts) are typed Gaussian instead of categorical.
- Parents and joins are encoded with the same attribute types as the children, so an attribute is continuous or discrete in the whole model.
- Declared Gaussian attributes with zero variance are ignored instead of making the posterior uniform.
- The feature-type parameters accept boolean masks.

### Fixes and quality

- Every estimator validates its parameters with scikit-learn's `_parameter_constraints`.
- The weighted models raise a single `ConvergenceWarning` when L-BFGS-B stops at `max_iter`.
- AnDE, TAN and KDB evaluate each component only at the augmented states of the classes when predicting, which cuts prediction time and memory.
- KDB raises a clear error when the parent states of an attribute exceed a 64-bit index.
- A brute-force reference implementation of AnDE checks the predictions of every order and pooling.
- The minimum supported versions are Python 3.10 and scikit-learn 1.6 (the earlier `scikit-learn>=1.4.2` requirement was wrong: the package needs 1.6). CI now also tests the minimum and the latest dependency versions.

## 1.2.1 (2026-10-13)

Release described in the PhD thesis of P. Torrijos: `MixedNB`, `AnDE`, `WeightedAnDE`, `AnJE`, `ALR`, `MDLPDiscretizer` and `DecisionTreeDiscretizer`.
