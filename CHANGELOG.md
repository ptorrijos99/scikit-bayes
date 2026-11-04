# Changelog

All notable changes to scikit-bayes are listed here. The project follows [semantic versioning](https://semver.org/).

## 1.3.0 (2026-11-04)

### New estimators

- `TAN`: Tree Augmented Naive Bayes, with a Chow-Liu tree on the conditional mutual information (contributed by Óscar Alcarria Laserna, #15).
- `KDB`: K-Dependence Bayesian classifier of Sahami (1996) (contributed by Óscar Alcarria Laserna, #16).
- `GaussianAnJE` and `GaussianALR`: AnJE and ALR built on the factors of the SPnDEs, so that continuous attributes keep Gaussian conditionals.
- `AnJE` and `ALR` now follow the definitions and training protocol of Zaidi et al. (2016), with the L-BFGS-B settings of their code (`factr = 1e7`, 5 corrections, no test on the projected gradient); `WeightedAnDE` gains weights per factor (`weight_scope="factor"`).

### Performance

- The SPnDEs of `AnDE`, `WeightedAnDE`, `GaussianAnJE` and `GaussianALR` are counted and evaluated by Numba kernels on flat tables that hold only the augmented states seen in training, instead of one `MixedNB` per component, and `AnDE` predicts without building the samples x classes x components tensor. The predictions do not change (beyond 1e-14); on Connect-4, A2DE fits 6 times and predicts 10 times faster.
- `AnJE` and `ALR` count and evaluate the n-joins with Numba kernels and a map from cells to table rows, instead of a matrix of cell keys and a sparse incidence matrix (4 GB for the folds of Covertype).
- `TAN` and `KDB` compute the conditional mutual information of every pair of attributes with a Numba kernel, which keeps the per-cell formula and its exact zeros, instead of a loop of NumPy calls per pair, and keep their conditionals in flat tables evaluated by a kernel instead of one `MixedNB` per attribute. Their cost was dominated by the d²/2 pairs: with 800 attributes TAN fits in 2.4 s instead of 30 s, and with 2,000 attributes and 125 samples (the shape of microarray data) in 4 s instead of 79 s.
- Categorical values that are integers in a small range are encoded with a lookup table instead of a sorted search, in every model.
- `n_jobs` sets the number of threads of the kernels; `AnJE`, `ALR`, `TAN` and `KDB` gain the parameter.
- The kernels are compiled without threads, and their parallel version only the first time a model runs with `n_jobs` other than 1. The compilation that the first use of a model pays after installing the package drops from 12-29 s to 4-15 s per model on a laptop; the compiled code is cached on disk. Large state grids are indexed with a radix sort instead of `np.unique`, whose compilation alone took 5 s.
- Numba is a new dependency.

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
- TAN and KDB raise a clear error when the parent states of an attribute exceed a 64-bit index.
- A brute-force reference implementation of AnDE checks the predictions of every order and pooling.
- The minimum supported versions are Python 3.10, scikit-learn 1.6 and Numba 0.58.1 (the earlier `scikit-learn>=1.4.2` requirement was wrong: the package needs 1.6). CI now also tests the minimum and the latest dependency versions.

## 1.2.1 (2026-10-13)

Release described in the PhD thesis of P. Torrijos: `MixedNB`, `AnDE`, `WeightedAnDE`, `AnJE`, `ALR`, `MDLPDiscretizer` and `DecisionTreeDiscretizer`.
