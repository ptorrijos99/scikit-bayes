"""AnDE against an independent brute-force implementation of its definition.

The reference recomputes every estimate from raw counts for each test
instance, as in the AODE and AnDE papers (Webb et al., 2005, 2012), with
Laplace estimates:

- P(y, x_P) = (#(y, x_P) + 1) / (t + |Y| |X_P|), smoothed over every
  augmented state, seen or not;
- P(x_j | y, x_P) = (#(x_j, y, x_P) + 1) / (#(y, x_P) + |X_j|), which is
  uniform for an augmented state never seen in training;
- a component takes part only when its parent instantiation occurred at
  least ``m`` times; with no component left, Naive Bayes is used.

It shares no code with the library, so it catches errors in the pooling, the
smoothing and the selection of components that tests built on the library's
own intermediate results cannot.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import InvalidParameterError

from skbn import ALR, TAN, AnDE, AnJE, MixedNB, WeightedAnDE

CARDS = np.array([3, 4, 3, 5, 3])


def _data(seed=42, n=150, n_test=300):
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.randint(0, c, n) for c in CARDS])
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.randint(0, 3, n)) % 3
    X_test = np.column_stack([rng.randint(0, c, n_test) for c in CARDS])
    return X, y, X_test


def _reference_ande(X, y, X_test, n_dependence, mean, min_count, cards=CARDS):
    classes = np.unique(y)
    n, n_features = X.shape
    out = np.zeros((len(X_test), len(classes)))
    for t, x in enumerate(X_test):
        joints = []
        for parents in combinations(range(n_features), n_dependence):
            parents = list(parents)
            match_p = np.all(X[:, parents] == x[parents], axis=1)
            if n_dependence and match_p.sum() < min_count:
                continue
            n_states = len(classes) * np.prod(cards[parents])
            row = np.zeros(len(classes))
            for c in classes:
                match_pc = match_p & (y == c)
                if n_dependence:
                    row[c] = np.log((match_pc.sum() + 1) / (n + n_states))
                else:  # Naive Bayes keeps the maximum-likelihood prior
                    row[c] = np.log(match_pc.sum() / n)
                for j in range(n_features):
                    if j not in parents:
                        hits = (X[match_pc, j] == x[j]).sum()
                        row[c] += np.log((hits + 1) / (match_pc.sum() + cards[j]))
            joints.append(row)
        assert joints, "every test instance keeps a component with these data"
        joints = np.array(joints)
        pooled = logsumexp(joints, axis=0) if mean == "arithmetic" else joints.mean(0)
        out[t] = pooled - logsumexp(pooled)
    return out


@pytest.mark.parametrize("n_dependence", [0, 1, 2])
@pytest.mark.parametrize("mean", ["arithmetic", "geometric"])
@pytest.mark.parametrize("min_count", [1, 3])
def test_ande_matches_brute_force_reference(n_dependence, mean, min_count):
    X, y, X_test = _data()
    model = AnDE(
        n_dependence=n_dependence,
        mean=mean,
        min_parent_count=min_count,
        categorical_features=list(range(X.shape[1])),
    ).fit(X, y)
    expected = _reference_ande(X, y, X_test, n_dependence, mean, min_count)
    assert_allclose(model.predict_log_proba(X_test), expected, atol=1e-10)


@pytest.mark.parametrize("mean", ["arithmetic", "geometric"])
def test_large_state_grids_match_brute_force_reference(mean):
    # Grids of augmented states with more than 4 cells per training sample
    # are indexed by sorting instead of with a dense array
    rng = np.random.RandomState(3)
    cards = [40, 35, 30, 3, 4]
    X = np.column_stack([rng.randint(0, c, 150) for c in cards])
    y = (X[:, 3] + X[:, 4] + rng.randint(0, 2, 150)) % 3
    seen = [np.unique(X[:, j]) for j in range(X.shape[1])]
    X_test = np.column_stack([rng.choice(values, 200) for values in seen])
    observed = np.array([len(values) for values in seen])
    assert 3 * observed[0] * observed[1] > 4 * len(X) + 1024
    model = AnDE(
        n_dependence=2, mean=mean, categorical_features=list(range(X.shape[1]))
    ).fit(X, y)
    expected = _reference_ande(X, y, X_test, 2, mean, 1, cards=observed)
    assert_allclose(model.predict_log_proba(X_test), expected, atol=1e-10)


def test_min_count_drops_components_for_some_instances():
    """Guard for the test above: with ``min_count=3`` some test instances lose
    some (but not all) of their components, so the selection is exercised."""
    X, _, X_test = _data()
    counts = np.array(
        [
            [
                np.sum(np.all(X[:, list(p)] == x[list(p)], axis=1))
                for p in combinations(range(5), 2)
            ]
            for x in X_test
        ]
    )
    n_active = (counts >= 3).sum(axis=1)
    assert (n_active < counts.shape[1]).any() and (n_active > 0).all()


ESTIMATORS = [MixedNB, AnDE, WeightedAnDE, AnJE, ALR, TAN]


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda e: e.__name__)
def test_invalid_parameters_are_rejected(estimator):
    X, y, _ = _data()
    with pytest.raises(InvalidParameterError, match="alpha"):
        estimator(alpha=-1.0).fit(X, y)


@pytest.mark.parametrize(
    "params",
    [
        {"n_dependence": -1},
        {"strategy": "median"},
        {"min_parent_count": -2},
        {"unseen_state": "zero"},
        {"var_prior_weight": -5.0},
    ],
)
def test_invalid_ande_parameters_are_rejected(params):
    X, y, _ = _data()
    with pytest.raises(InvalidParameterError, match=next(iter(params))):
        AnDE(**params).fit(X, y)


def test_n_dependence_larger_than_n_features_is_rejected():
    X, y, _ = _data()
    for estimator in (AnDE, WeightedAnDE):
        with pytest.raises(ValueError, match="n_dependence"):
            estimator(n_dependence=X.shape[1] + 1).fit(X, y)


@pytest.mark.parametrize("modular", [False, True])
def test_unconverged_weights_warn_once(modular):
    X, y, _ = _data()
    with pytest.warns(ConvergenceWarning, match="max_iter=1") as record:
        WeightedAnDE(weight_level=4, max_iter=1, modular=modular).fit(X, y)
    assert sum(issubclass(w.category, ConvergenceWarning) for w in record) == 1
    with pytest.warns(ConvergenceWarning):
        ALR(max_iter=1).fit(X, y)
