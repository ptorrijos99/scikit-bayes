# Changelog

All notable changes to scikit-bayes are listed here. The project follows [semantic versioning](https://semver.org/).

## Unreleased

### New

- `strategy="mdlp"` in every model that discretizes (`AnDE`, `WeightedAnDE`, `AnJE`, `ALR`, `GaussianAnJE`, `GaussianALR`, `TAN`, `KDB`): the continuous attributes are cut with the entropy discretization of Fayyad and Irani instead of `KBinsDiscretizer`, wherever the model discretizes them (every continuous attribute in `AnJE` and `ALR`; the super-parents and the structure in the others, whose continuous children stay Gaussian). It needs no pipeline and no declaration of types, and reproduces the discretization of the AnDE, ALR and TAN papers.

### Changed

- `AnJE` and `ALR`: a value never seen in training no longer falls in an empty cell of the n-joins that contain it; those joins are left out for that sample, as the other models leave out the attribute. The estimate of an empty cell, `(m / |X_alpha|) / (#y + m)`, depends on the class only through its size, so it multiplied the posterior by the inverse of the class counts: once in `AnJE`, which cancelled the class prior, and once per join in `ALR`. Predictions change only for samples with unseen values.
- `MDLPDiscretizer(max_bins=...)` keeps the cuts that remove the most class entropy. It used to keep the first ones its depth-first search reached, which could all lie at one end of the range. Without `max_bins` the cut-points are the same.
- The models no longer let through the warning of `KBinsDiscretizer` about bins that are too narrow, which named the wrong attribute; the number of bins left is in the fitted model as before.

### Fixed

- The user guide said that the codes of a discretizer in a pipeline are typed as categorical. With more than 20 intervals, which `MDLPDiscretizer` produces on large samples, they are typed as continuous and discretized again unless declared in `categorical_features`. The guide now says so and recommends `strategy="mdlp"`.
- The docstring of `ALR` said that `w = 1` is the `AnJE`; it is the unaveraged product of the n-joins, as the user guide states.

### Performance

- `TAN` and `KDB` on attributes of thousands of values: the conditional mutual information of a pair whose (class, value, value) table has more than four cells per training sample sums over its non-empty cells only, and a conditional whose (state, value) table would be that large keeps only its non-empty cells. The probabilities, the structure and the tables are the same as before. On one fold of KDD Cup 09 Upselling (11 attributes of more than 100 values, up to 13 333), with one CPU, TAN fits in 5.2 s instead of 101 s and peaks at 0.68 GB instead of 35 GB, and KDB with `k=2` fits in 5.6 s instead of 124 s and peaks at 0.70 GB instead of 52 GB. On Amazon Employee Access, TAN peaks at 0.31 GB instead of 2.7 GB. Data without such attributes fit as before.

## 1.3.2 (2026-11-06)

`TAN` and `KDB` no longer run out of memory on attributes with thousands of values. The API is unchanged and every model predicts exactly as in 1.3.1.

### Performance

- `TAN` and `KDB` store the tables of attributes with very many values sparsely:
  - The conditional mutual information of a pair whose table would have more than four cells per training sample is summed over its non-empty cells only, in the order of the dense loop, so the values and the learned structure are the same.
  - A conditional table of that size keeps its non-empty cells and, for each parent state, the probability of an unseen value; prediction finds a cell by binary search.
  - The log-probability tables are built one factor at a time.
  - On KDD Cup 09 Upselling (attributes with up to 13 333 values; one fold, one CPU), TAN fits in 5.2 s instead of 101 s and peaks at 0.68 GB instead of 35 GB, and KDB with k=2 in 5.6 s instead of 124 s and 0.70 GB instead of 52 GB. Datasets of ordinary cardinality keep their times and memory.

### Documentation

- The discretizers have their API pages and a section of the user guide, and the docs show the released version instead of a development one.
- The gallery figures share one style, with a fixed colour per model, and the timed examples exclude the compilation of the kernels.
- New examples: AnJE against ALR with and without penalty as the sample grows, Gaussian children against discretized attributes, and AnJE after MDLP, tree and quantile discretization.
- The README, the user guide and the text of the examples are rewritten in plainer prose.

## 1.3.1 (2026-11-06)

A performance release: the API is unchanged and, on x86-64, every model predicts exactly as in 1.3.0. The times below are from one fold of the benchmark datasets on a laptop, comparing 1.3.0 and 1.3.1 in the same process.

### Performance

- `AnDE`:
  - Components whose parents differ only in constant attributes share their tables, so they are counted and scored once.
  - A1DE counts the children of each pair of attributes once, and A2DE those of each triple, instead of once per component.
  - Augmented states seen in at most 16 training rows keep those rows instead of a dense row of counts.
  - The training codes are kept in narrow integers, and the categorical attributes are encoded once per fit.
  - Prediction looks up the states of every class first and then reads each child once.
  - On Covertype, A1DE fits in 2.1 s instead of 8.4 s, predicts in 0.29 s instead of 9.2 s, and peaks at 0.70 GB instead of 1.26 GB. On Census, A2DE fits in 8.6 s instead of 36 s and predicts in 0.37 s instead of 8.9 s. A2DE on First-order theorem proving peaks at 0.63 GB instead of 2.5 GB, and A1DE predicts Bioresponse in 0.33 s instead of 290 s.
- `WeightedAnDE`: the objective and its gradient are Numba kernels for both means and both weight scopes, in place of sparse products and samples x classes x components arrays. The kernels add in the same order as NumPy and SciPy did. With the default settings it fits Nomao in 27 s instead of 159 s and Census in 79 s instead of 255 s, and predicts about 20 times faster.
- `AnJE` and `ALR`:
  - The n-joins are counted in one pass, and the prediction buffer is bounded.
  - The two-class ALR objective scores four samples at a time.
  - A2JE fits Covertype in 2.3 s instead of 9.4 s, and fits GINA in 3.7 s instead of 17 s using 0.41 GB instead of 1.26 GB. ALR fits about 3 times faster.
- `TAN` and `KDB` count the attribute pairs of the conditional mutual information as integers and fit about twice as fast (TAN on Covertype: 3.8 s instead of 7.6 s).
- Importing `skbn` no longer loads `sklearn.tree` and `sklearn.impute`, which only the discretizers use. The import now adds 5 MB to scikit-learn and Numba instead of 15 MB.

### Numerical note

The weighted models now compute in Numba arithmetic that 1.3.0 left to NumPy and SciPy, so a few platforms can show rounding-level differences:

- The arithmetic mean of `WeightedAnDE` reproduces the log-sum-exp of SciPy 1.15 and later. With an older SciPy, its fitted weights can differ from 1.3.0 by rounding, as 1.3.0 already differed between SciPy versions.
- On ARM processors, including Apple silicon, SciPy's sparse product fuses multiplications and additions and the kernels do not, so the geometric objective can differ in the last bit.

### Development

- The gradient tests compare directional derivatives along a few random directions instead of approximating every partial derivative.
- The scikit-learn API checks run with a few optimizer iterations.
- With these changes the test suite runs in about a minute, and the coverage job, which runs the kernels as Python, in about four.

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
