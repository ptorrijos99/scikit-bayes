"""Tests for WeightedTAN and WeightedKDB: the generative model at w = 1, the
weight layout, the objective and its gradient, the match with the factor
weights of WeightedAnDE on Naive Bayes, and what the weights learn."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss

from skbn import KDB, TAN, WeightedAnDE, WeightedKDB, WeightedTAN

from ._gradient import assert_gradient

LEVELS = [1, 2, 3, 4]
TYPES = dict(
    gaussian_features=[0, 1], categorical_features=[2, 3], bernoulli_features=[4]
)
# The checks of the fit stop the optimizer early on purpose
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

NETWORKS = [
    pytest.param(TAN, WeightedTAN, {"root": 3}, id="TAN"),
    pytest.param(KDB, WeightedKDB, {"k": 2}, id="KDB2"),
]


def _mixed_data(n_classes, n=300, seed=0):
    """Gaussian (0, 1), categorical (2, 3) and Bernoulli (4) attributes.
    Category 3 of attribute 3 occurs only with the last class, so that the
    states of the other classes with that parent value are never seen."""
    rng = np.random.RandomState(seed)
    X_cont = rng.normal(size=(n, 2))
    X_cont[:, 1] += X_cont[:, 0]
    X_cat = rng.randint(0, 4, size=(n, 2))
    X_bin = rng.randint(0, 2, size=(n, 1))
    X = np.hstack([X_cont, X_cat, X_bin]).astype(float)
    score = X_cont[:, 0] + X_cat[:, 0] + X_bin[:, 0] + rng.normal(scale=0.5, size=n)
    y = (score > 2.0).astype(int)
    if n_classes > 2:
        y[score > 3.5] = 2
    y[X_cat[:, 1] == 3] = n_classes - 1
    return X, y


def _test_rows(X):
    """Training rows, then rows with a category never seen (attribute 2), a
    parent value never seen (category 3 in attribute 3, of every class but
    the last) and extreme continuous values."""
    rng = np.random.RandomState(1)
    rows = X[rng.choice(len(X), 60)].copy()
    rows[:10, 2] = 9.0
    rows[10:20, 3] = 3.0
    rows[20:25, 3] = 7.0
    rows[25:30, 0] = 50.0
    rows[30:35, 1] = -50.0
    return np.vstack([X, rows])


def _training_design(model, X, y):
    """Log-terms, the weight of every (sample, class, term) and the one-hot
    targets. The design keeps the weight of class 0 of each term; those of
    the other classes follow it, and the untrained weight is its own for
    every class."""
    terms, base = model._design(model._evaluation_inputs(X))
    n_classes, untrained = terms.shape[1], model.n_weights_ - 1
    step = np.arange(n_classes)[:, None] * (model._class_weights > 1)
    index = np.where(base[:, None] == untrained, untrained, base[:, None] + step)
    return terms, index, np.eye(n_classes)[np.searchsorted(model.classes_, y)]


def _parents(model):
    """Attribute parents of every attribute, as lists (TAN marks the root
    with -1)."""
    return [[int(p) for p in np.atleast_1d(ps) if p >= 0] for ps in model.parents_]


# --- The generative model at w = 1 -----------------------------------------


@pytest.mark.parametrize("Generative, Weighted, params", NETWORKS)
@pytest.mark.parametrize("level", LEVELS)
def test_unit_weights_reproduce_the_generative_model(
    Generative, Weighted, params, level
):
    X, y = _mixed_data(3)
    X_test = _test_rows(X)
    generative = Generative(**params, **TYPES).fit(X, y)
    model = Weighted(**params, weight_level=level, max_iter=5, **TYPES).fit(X, y)
    assert _parents(model) == _parents(generative)

    # The test rows reach states never seen in training
    inputs, t = model._evaluation_inputs(X_test), model._tables
    n_unseen = 0
    for m in np.flatnonzero(t.n_parents):
        p = t.parents[m, : t.n_parents[m]]
        key = inputs.pcodes[:, p] @ t.strides[m, : t.n_parents[m]]
        seen = t.state_keys[t.state_off[m] : t.state_off[m + 1]]
        for c in range(3):
            n_unseen += np.sum(~np.isin(c * t.n_combos[m] + key, seen))
        n_unseen += inputs.punseen[:, p].any(axis=1).sum()
    assert n_unseen > 0

    model.learned_weights_ = np.ones(model.n_weights_)
    assert_allclose(
        model.predict_proba(X_test),
        generative.predict_proba(X_test),
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(
        model._joint_log_likelihood(X_test),
        generative._joint_log_likelihood(X_test),
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("Generative, Weighted, params", NETWORKS)
def test_weighted_scores_are_the_weighted_sum_of_the_terms(
    Generative, Weighted, params
):
    # Every term is a log-probability of the generative network (they sum to
    # its joint) and the score is the sum of the terms times their weights
    X, y = _mixed_data(3)
    X_test = _test_rows(X)
    model = Weighted(**params, max_iter=5, **TYPES).fit(X, y)
    terms, index, _ = _training_design(model, X_test, np.zeros(len(X_test)))
    generative = Generative(**params, **TYPES).fit(X, y)
    assert_allclose(terms.sum(axis=2), generative._joint_log_likelihood(X_test))
    assert_allclose(
        terms[:, :, 0],
        np.broadcast_to(model.backoff_.class_log_prior_, terms.shape[:2]),
    )
    w = np.random.RandomState(0).uniform(0.5, 1.5, model.n_weights_)
    model.learned_weights_ = w
    assert_allclose(
        model._joint_log_likelihood(X_test), (w[index] * terms).sum(axis=2), atol=1e-10
    )


# --- Weight layout ------------------------------------------------------------


@pytest.mark.parametrize("Generative, Weighted, params", NETWORKS)
def test_weight_counts(Generative, Weighted, params):
    # Class prior and one factor per attribute; the categorical children have
    # 4 values, the Bernoulli one 2 and the Gaussian ones a single
    # (undiscretized) value. Only the combinations seen in training get a
    # weight, plus one for the rest.
    X, y = _mixed_data(3)
    model = Weighted(**params, max_iter=1, **TYPES)
    counts = {}
    for level in LEVELS:
        with pytest.warns(ConvergenceWarning):
            counts[level] = model.set_params(weight_level=level).fit(X, y).n_weights_
    values = [1, 1, 4, 4, 2]
    assert counts[1] == 1 + 5 + 1
    assert counts[2] == 1 + sum(values) + 1
    assert counts[3] == 3 * (counts[2] - 1) + 1

    # Level 4: the (parent values, child value) combinations that occur, by
    # class; the prior has one weight per class
    codes = model._encoder.transform(X)[0]
    expected = 3 + 1
    for i, parents in enumerate(_parents(model)):
        child = X[:, [i]] if values[i] > 1 else np.zeros((len(X), 1))
        expected += 3 * len(np.unique(np.hstack([codes[:, parents], child]), axis=0))
    assert counts[4] == expected


def test_unseen_parent_value_takes_the_untrained_weight():
    X, y = _mixed_data(3)
    model = WeightedKDB(k=2, max_iter=5, **TYPES).fit(X, y)
    X_test = X[:5].copy()
    X_test[:, 3] = 7.0  # never seen
    index = _training_design(model, X_test, np.zeros(len(X_test)))[1]
    children = model._tables.children
    has_parent_3 = [1 + m for m, i in enumerate(children) if 3 in model.parents_[i]]
    assert has_parent_3
    assert np.all(index[:, :, has_parent_3] == model.n_weights_ - 1)
    assert model.learned_weights_[-1] == 1.0


# --- Objective and gradient ---------------------------------------------------


@pytest.mark.parametrize("Generative, Weighted, params", NETWORKS)
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_gradient_matches_finite_differences(
    Generative, Weighted, params, level, n_classes
):
    X, y = _mixed_data(n_classes)
    model = Weighted(**params, weight_level=level, max_iter=1, **TYPES)
    with pytest.warns(ConvergenceWarning):
        model.fit(X, y)
    terms, index, Y = _training_design(model, X, y)
    w = np.random.RandomState(1).uniform(0.4, 1.6, model.n_weights_)
    base = model._design(model._evaluation_inputs(X))[1]
    nll, grad = model._nll_and_grad(w, terms, base, Y)

    # The objective in array form
    scores = (w[index] * terms).sum(axis=2)
    expected = -np.sum((scores - logsumexp(scores, axis=1, keepdims=True)) * Y)
    assert_allclose(nll, expected, rtol=1e-12)
    assert_gradient(lambda t: model._nll_and_grad(t, terms, base, Y)[0], w, grad)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_weighted_naive_bayes_matches_factor_weighted_ande(level, n_classes):
    # KDB with k=0 is Naive Bayes, the single SPnDE of AnDE with
    # n_dependence=0: its factor weights in the geometric pool are the same
    # model, with the same weights in the same order. The two differ in the
    # Gaussian variances unless var_prior_weight=0: the component of AnDE
    # without parents does not shrink them, MixedNB (and so KDB) does.
    X, y = _mixed_data(n_classes)
    params = dict(weight_level=level, l2_reg=0.5, var_prior_weight=0.0, **TYPES)
    kdb = WeightedKDB(k=0, max_iter=30, **params).fit(X, y)
    ande = WeightedAnDE(
        n_dependence=0, weight_scope="factor", mean="geometric", max_iter=30, **params
    ).fit(X, y)
    assert kdb.n_weights_ == ande.n_weights_

    terms, _, Y = _training_design(kdb, X, y)
    index = kdb._design(kdb._evaluation_inputs(X))[1]
    design, _ = ande._design(X)
    w = np.random.RandomState(2).uniform(0.4, 1.6, kdb.n_weights_)
    nll, grad = kdb._nll_and_grad(w, terms, index, Y)
    ande_nll, ande_grad = ande._nll_and_grad(w, design, Y)
    assert_allclose(nll, ande_nll, rtol=1e-12)
    assert_allclose(grad, ande_grad, rtol=1e-9, atol=1e-10)

    assert_allclose(kdb.learned_weights_, ande.learned_weights_, atol=1e-6)
    X_test = _test_rows(X)
    assert_allclose(kdb.predict_proba(X_test), ande.predict_proba(X_test), atol=1e-6)


# --- Fit --------------------------------------------------------------------


def _duplicated_data(n, seed):
    """A Gaussian attribute five times, and three others, all informative."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    x = rng.normal(size=n) + 0.8 * y
    other = rng.normal(size=(n, 3)) + 0.5 * y[:, None]
    return np.column_stack([np.repeat(x[:, None], 5, axis=1), other]), y


@pytest.mark.parametrize("Generative, Weighted, params", NETWORKS)
@pytest.mark.parametrize("level", [1, 4])
def test_weights_correct_a_duplicated_attribute(Generative, Weighted, params, level):
    # A continuous child is Gaussian given the bin of its parent, so its
    # copies still count its evidence again: the weights discount them
    X, y = _duplicated_data(600, 0)
    X_test, y_test = _duplicated_data(4000, 1)
    generative = Generative(**params).fit(X, y)
    model = Weighted(**params, weight_level=level).fit(X, y)
    before = log_loss(y_test, generative.predict_proba(X_test))
    after = log_loss(y_test, model.predict_proba(X_test))
    assert after < before - 0.015


def test_regularization_pulls_weights_towards_one():
    X, y = _duplicated_data(600, 0)

    def deviation(l2_reg):
        model = WeightedTAN(weight_level=3, l2_reg=l2_reg).fit(X, y)
        return np.abs(model.learned_weights_ - 1.0).mean()

    assert deviation(100.0) < 0.1 * deviation(1e-3)


def test_too_few_iterations_warn():
    X, y = _mixed_data(3)
    with pytest.warns(ConvergenceWarning, match="max_iter=1 before converging;"):
        model = WeightedTAN(max_iter=1, **TYPES).fit(X, y)
    assert model.n_iter_ == 1


@pytest.mark.parametrize("Generative, Weighted, params", NETWORKS)
def test_threads_give_the_same_model(Generative, Weighted, params):
    X, y = _mixed_data(3)
    X_test = _test_rows(X)
    serial = Weighted(**params, n_jobs=1, max_iter=30, **TYPES).fit(X, y)
    threaded = Weighted(**params, n_jobs=2, max_iter=30, **TYPES).fit(X, y)
    assert_array_equal(threaded.learned_weights_, serial.learned_weights_)
    assert_array_equal(threaded.predict_proba(X_test), serial.predict_proba(X_test))


@pytest.mark.parametrize("level", [0, 5])
def test_invalid_weight_level_raises(level):
    X, y = _mixed_data(2)
    with pytest.raises(ValueError, match="weight_level"):
        WeightedKDB(weight_level=level).fit(X, y)
