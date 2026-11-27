"""Tests for sample_weight support across every classifier."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB

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
    WeightedNB(max_iter=30),
    AnDE(n_dependence=1),
    AnDE(n_dependence=2),
    AnDE(n_dependence=1, strategy="mdlp", continuous_children="discretized"),
    WeightedAnDE(max_iter=30),
    WeightedAnDE(weight_scope="factor", weight_level=3, max_iter=30),
    AnJE(n_join=2),
    ALR(n_join=2, l2_reg=1.0, max_iter=30),
    GaussianAnJE(),
    GaussianALR(l2_reg=1.0, max_iter=30),
    TAN(),
    KDB(k=2),
    WeightedTAN(max_iter=30),
    WeightedKDB(k=2, max_iter=30),
]


def _mixed_data(n=120, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=n)
    x_cont = rng.normal(size=(n, 2)) + y[:, None]
    x_cat = (rng.randint(0, 3, size=(n, 1)) + y[:, None]) % 3
    x_bin = ((rng.rand(n, 1) < 0.7) ^ (y[:, None] == 0)).astype(float)
    X = np.hstack([x_cont, x_cat, x_bin])
    return X, y


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_unit_sample_weight_matches_unweighted(model):
    X, y = _mixed_data()
    unweighted = clone(model).fit(X, y)
    weighted = clone(model).fit(X, y, sample_weight=np.ones(len(y)))
    assert_allclose(
        weighted.predict_proba(X), unweighted.predict_proba(X), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_integer_sample_weight_with_zeros_matches_row_repetition(model):
    X, y = _mixed_data(n=90, seed=1)
    rng = np.random.RandomState(2)
    sw = rng.randint(0, 4, size=len(y))
    # Guarantee every class keeps positive weight
    for c in range(3):
        sw[np.flatnonzero(y == c)[0]] = 2

    X_rep = np.repeat(X, sw, axis=0)
    y_rep = np.repeat(y, sw)

    m_sw = clone(model).fit(X, y, sample_weight=sw)
    m_rep = clone(model).fit(X_rep, y_rep)
    assert_allclose(m_sw.predict_proba(X), m_rep.predict_proba(X), rtol=1e-4, atol=1e-5)


def test_mixed_nb_sample_weight_matches_sklearn_gaussian_nb():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(150, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    sw = rng.uniform(0.1, 3.0, size=len(y))
    ours = MixedNB().fit(X, y, sample_weight=sw).predict_proba(X)
    theirs = GaussianNB().fit(X, y, sample_weight=sw).predict_proba(X)
    assert_allclose(ours, theirs, rtol=1e-7, atol=1e-9)


def test_invalid_sample_weights_are_rejected():
    X, y = _mixed_data(n=30)
    with pytest.raises(ValueError, match="non-negative"):
        MixedNB().fit(X, y, sample_weight=np.full(len(y), -1.0))
    # Recent scikit-learn rejects all-zero weights itself, with its own message
    with pytest.raises(ValueError, match="positive sum|at least one non-zero"):
        TAN().fit(X, y, sample_weight=np.zeros(len(y)))
    with pytest.raises(ValueError):
        AnDE().fit(X, y, sample_weight=np.ones(len(y) - 1))


@pytest.mark.parametrize(
    "model",
    [
        MixedNB(),
        WeightedNB(max_iter=15),
        AnDE(),
        WeightedAnDE(max_iter=15),
        TAN(),
        WeightedTAN(max_iter=15),
        AnJE(),
        ALR(l2_reg=1.0, max_iter=15),
    ],
    ids=repr,
)
def test_zero_weight_class_produces_finite_probabilities(model):
    """A class whose samples all have weight 0 does not produce NaN or inf."""
    X, y = _mixed_data(n=90, seed=4)
    sw = np.where(y == 2, 0.0, 1.0)
    proba = clone(model).fit(X, y, sample_weight=sw).predict_proba(X[:20])
    assert proba.shape == (20, 3)
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
