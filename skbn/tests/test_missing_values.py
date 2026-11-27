"""Tests for native missing-value (NaN / None) handling across every classifier."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skbn import (
    ALR,
    KDB,
    TAN,
    AnDE,
    AnJE,
    GaussianALR,
    GaussianAnJE,
    MixedNB,
    WeightedAnDE,
    WeightedKDB,
    WeightedNB,
    WeightedTAN,
)

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

CLASSIFIERS = [
    MixedNB(),
    WeightedNB(max_iter=20),
    AnDE(n_dependence=1),
    AnDE(n_dependence=2),
    AnDE(n_dependence=1, strategy="mdlp", continuous_children="discretized"),
    WeightedAnDE(max_iter=20),
    WeightedAnDE(weight_scope="factor", weight_level=3, max_iter=20),
    AnJE(n_join=2),
    ALR(n_join=2, l2_reg=1.0, max_iter=20),
    GaussianAnJE(),
    GaussianALR(l2_reg=1.0, max_iter=20),
    TAN(),
    KDB(k=2),
    WeightedTAN(max_iter=20),
    WeightedKDB(k=2, max_iter=20),
]


def _mixed_with_nans(n=150, seed=0, missing_rate=0.15):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=n)
    x_cont = rng.normal(size=(n, 2)) + y[:, None]
    x_cat = ((rng.randint(0, 3, size=(n, 1)) + y[:, None]) % 3).astype(float)
    x_bin = ((rng.rand(n, 1) < 0.7) ^ (y[:, None] == 0)).astype(float)
    X = np.hstack([x_cont, x_cat, x_bin])
    mask = rng.rand(*X.shape) < missing_rate
    X[mask] = np.nan
    return X, y


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_fit_and_predict_with_missing_values(model):
    X_train, y_train = _mixed_with_nans(n=160, seed=0)
    X_test, _ = _mixed_with_nans(n=40, seed=1)
    # Include an all-NaN test row
    X_test[0] = np.nan

    fitted = clone(model).fit(X_train, y_train)
    proba = fitted.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    preds = fitted.predict(X_test)
    assert set(preds).issubset(set(fitted.classes_))


@pytest.mark.parametrize("col", [0, 1, 2])
def test_mixed_nb_marginalises_missing_feature_at_predict(col):
    """Setting feature ``col`` to NaN at prediction time matches fitting
    MixedNB without that feature, for Gaussian, categorical and Bernoulli."""
    rng = np.random.RandomState(0)
    n = 200
    y = rng.randint(0, 2, size=n)
    X = np.column_stack(
        [
            rng.normal(size=n) + y,  # Gaussian
            (rng.randint(0, 3, size=n) + y) % 3,  # Categorical
            (rng.rand(n) < (0.8 * y + 0.1)).astype(float),  # Bernoulli
        ]
    )
    full = MixedNB(
        gaussian_features=[0], categorical_features=[1], bernoulli_features=[2]
    ).fit(X, y)
    keep = [j for j in range(3) if j != col]
    reduced = MixedNB(
        gaussian_features=[j for j in [0] if j != col] if col != 0 else None,
        categorical_features=[keep.index(1)] if col != 1 else None,
        bernoulli_features=[keep.index(2)] if col != 2 else None,
    ).fit(X[:, keep], y)

    X_missing = X[:30].copy()
    X_missing[:, col] = np.nan
    assert_allclose(
        full.predict_proba(X_missing),
        reduced.predict_proba(X[:30, keep]),
        atol=1e-12,
    )


def test_all_nan_test_row_returns_class_prior():
    rng = np.random.RandomState(4)
    y = np.array([0] * 60 + [1] * 140)
    X = np.column_stack([rng.normal(size=200) + y, rng.randint(0, 3, size=200)])
    prior = np.array([0.3, 0.7])
    for model in (MixedNB(), AnDE(n_dependence=1), TAN(), KDB(k=1)):
        proba = model.fit(X, y).predict_proba(np.full((1, 2), np.nan))
        assert_allclose(proba[0], prior, atol=1e-12)


@pytest.mark.parametrize("model", [MixedNB(), AnDE(), TAN(), KDB(), AnJE()], ids=repr)
def test_all_nan_training_column_is_ignored(model):
    rng = np.random.RandomState(5)
    y = rng.randint(0, 2, size=100)
    X = np.column_stack([rng.normal(size=100) + y, np.full(100, np.nan)])
    proba = clone(model).fit(X, y).predict_proba(X[:10])
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "model",
    [MixedNB(), WeightedNB(max_iter=15), AnDE(), TAN(), ALR(l2_reg=1.0, max_iter=15)],
    ids=repr,
)
def test_missing_values_combined_with_sample_weight(model):
    X, y = _mixed_with_nans(n=120, seed=6)
    rng = np.random.RandomState(7)
    sw = rng.randint(0, 3, size=len(y))
    for c in range(3):
        sw[np.flatnonzero(y == c)[0]] = 1
    X_rep = np.repeat(X, sw, axis=0)
    y_rep = np.repeat(y, sw)

    p_sw = clone(model).fit(X, y, sample_weight=sw).predict_proba(X)
    p_rep = clone(model).fit(X_rep, y_rep).predict_proba(X)
    assert_allclose(p_sw, p_rep, rtol=1e-4, atol=1e-5)


def test_inf_is_still_rejected():
    X = np.array([[1.0, 2.0], [np.inf, 1.0], [0.0, 3.0], [2.0, 0.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="infinity"):
        MixedNB().fit(X, y)


def test_all_nan_and_constant_columns_do_not_merge_in_a2de():
    """An all-NaN column must not be grouped with a constant column in A2DE."""
    rng = np.random.RandomState(8)
    n = 120
    y = rng.randint(0, 2, size=n)
    x_inf = rng.normal(size=n) + 2.0 * y
    X = np.column_stack([np.full(n, np.nan), np.zeros(n), x_inf])
    clf = AnDE(n_dependence=2).fit(X, y)
    proba = clf.predict_proba(X)
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0)
    # Component (1, 2) (constant + informative) must stay active and accurate
    assert clf.score(X, y) > 0.8


@pytest.mark.parametrize(
    "model",
    [TAN(), KDB(k=1), WeightedTAN(max_iter=10), WeightedKDB(k=1, max_iter=10)],
    ids=repr,
)
def test_disjoint_missing_columns_in_networks(model):
    """Two attributes never jointly observed produce valid predictions."""
    rng = np.random.RandomState(9)
    n = 80
    y = rng.randint(0, 2, size=n)
    x0 = np.where(np.arange(n) < 40, rng.normal(size=n) + y, np.nan)
    x1 = np.where(np.arange(n) >= 40, rng.normal(size=n) + y, np.nan)
    X = np.column_stack([x0, x1])
    proba = clone(model).fit(X, y).predict_proba(X)
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0)


def test_anje_alpha_zero_with_missing_values():
    X, y = _mixed_with_nans(n=100, seed=10, missing_rate=0.3)
    proba = AnJE(n_join=2, alpha=0.0).fit(X, y).predict_proba(X[:20])
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0)
