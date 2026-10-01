"""Tests for the estimation details of the AnDE family: aggregation rules,
zero-count handling, variance shrinkage and analytic gradients."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB

from skbn import ALR, AnDE, AnJE, MixedNB, WeightedAnDE


@pytest.fixture
def mixed_data():
    rng = np.random.RandomState(0)
    n = 300
    X_cont = rng.normal(size=(n, 3))
    X_cat = rng.randint(0, 4, size=(n, 2)).astype(float)
    X = np.hstack([X_cont, X_cat])
    y = ((X_cont[:, 0] + X_cat[:, 0] + rng.normal(scale=0.5, size=n)) > 1.5).astype(int)
    y[X_cat[:, 1] == 3] = 2
    return X, y


def _fit_kwargs():
    return dict(gaussian_features=[0, 1, 2], categorical_features=[3, 4])


# --- Aggregation rules ------------------------------------------------------


def test_anje_is_geometric_mean(mixed_data):
    X, y = mixed_data
    model = AnJE(n_dependence=1, **_fit_kwargs()).fit(X, y)
    jll, _, active, _ = model._get_jll_per_model(X)
    assert active.all()
    expected = jll.mean(axis=2)
    expected -= logsumexp(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(model.predict_log_proba(X), expected, atol=1e-10)


def test_ande_is_arithmetic_mean(mixed_data):
    X, y = mixed_data
    model = AnDE(n_dependence=1, **_fit_kwargs()).fit(X, y)
    jll, _, active, _ = model._get_jll_per_model(X)
    expected = logsumexp(jll, axis=2)
    expected -= logsumexp(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(model.predict_log_proba(X), expected, atol=1e-10)


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_alr_with_unit_weights_is_anje(mixed_data, level):
    X, y = mixed_data
    anje = AnJE(n_dependence=1, **_fit_kwargs()).fit(X, y)
    alr = ALR(n_dependence=1, weight_level=level, max_iter=50, **_fit_kwargs()).fit(
        X, y
    )
    alr.coef_theta_ = np.ones_like(alr.coef_theta_)
    np.testing.assert_allclose(alr.predict_proba(X), anje.predict_proba(X), atol=1e-10)


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_wande_with_unit_weights_is_ande(mixed_data, level):
    X, y = mixed_data
    ande = AnDE(n_dependence=1, **_fit_kwargs()).fit(X, y)
    wande = WeightedAnDE(
        n_dependence=1, weight_level=level, max_iter=50, **_fit_kwargs()
    ).fit(X, y)
    wande.coef_theta_ = np.zeros_like(wande.coef_theta_)
    np.testing.assert_allclose(
        wande.predict_proba(X), ande.predict_proba(X), atol=1e-10
    )


@pytest.mark.parametrize("level", [1, 2])
def test_modular_wande_class_independent_levels_equal_ande(mixed_data, level):
    X, y = mixed_data
    ande = AnDE(n_dependence=1, **_fit_kwargs()).fit(X, y)
    with pytest.warns(UserWarning, match="equivalent to the generative AnDE"):
        wande = WeightedAnDE(
            n_dependence=1, weight_level=level, modular=True, **_fit_kwargs()
        ).fit(X, y)
    np.testing.assert_allclose(wande.learned_weights_, 1.0)
    np.testing.assert_allclose(
        wande.predict_proba(X), ande.predict_proba(X), atol=1e-10
    )


@pytest.mark.parametrize("level", [3, 4])
def test_modular_wande_class_specific_levels_learn(mixed_data, level):
    X, y = mixed_data
    wande = WeightedAnDE(
        n_dependence=1, weight_level=level, modular=True, **_fit_kwargs()
    ).fit(X, y)
    assert not np.allclose(wande.learned_weights_, 1.0)


# --- Analytic gradients -----------------------------------------------------


@pytest.mark.parametrize("cls", [ALR, WeightedAnDE])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_joint_gradient_matches_finite_differences(mixed_data, cls, level):
    X, y = mixed_data
    model = cls(
        n_dependence=1, weight_level=level, max_iter=1, l2_reg=0.1, **_fit_kwargs()
    ).fit(X, y)
    jll, _, active, _ = model._get_jll_per_model(X)
    idx = model._expand_indices(model._w_indices, len(model.classes_))
    Y = np.eye(len(model.classes_))[np.searchsorted(model.classes_, y)]
    rng = np.random.RandomState(1)
    theta = model.coef_theta_ + rng.uniform(-0.3, 0.3, size=model.n_weights_)
    if cls is ALR:
        theta = np.abs(theta)
    _, grad = model._objective(theta, jll, active, idx, Y)
    num = approx_fprime(
        theta, lambda t: model._objective(t, jll, active, idx, Y)[0], 1e-6
    )
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-3)


# --- Zero counts ------------------------------------------------------------


def test_unseen_augmented_state_backs_off_instead_of_vetoing():
    # Parent value 1 never co-occurs with class 1 in training.
    X = np.array([[0, 0.1], [0, 0.2], [1, 0.3], [1, 0.4], [0, 2.1], [0, 2.2], [0, 2.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    model = AnJE(
        n_dependence=1,
        categorical_features=[0],
        gaussian_features=[1],
        unseen_state="backoff",
    ).fit(X, y)
    jll, _, active, _ = model._get_jll_per_model(np.array([[1, 2.1]]))
    assert np.all(np.isfinite(jll))
    assert jll.min() > -100  # no hard -700 floor
    proba = model.predict_proba(np.array([[1, 2.1]]))
    assert proba[0, 1] > 0.5  # the continuous child still points to class 1


def test_unseen_parent_value_excludes_component_and_falls_back_to_nb():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    model = AnDE(n_dependence=2, categorical_features=[0, 1]).fit(X, y)
    # Remove the only pair SPnDE's knowledge of (1, 1) by querying a value mapped
    # to a known code but with min_parent_count above the training frequency.
    model.min_parent_count = 1000
    jll, _, active, nb_jll = model._get_jll_per_model(X[:3])
    assert not active.any()
    nb = MixedNB(categorical_features=[0, 1]).fit(X, y)
    np.testing.assert_allclose(
        model.predict_proba(X[:3]), nb.predict_proba(X[:3]), atol=1e-6
    )


# --- Variance shrinkage -----------------------------------------------------


def test_mixednb_zero_prior_weight_matches_gaussiannb():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    y = rng.randint(0, 2, size=50)
    ours = MixedNB(gaussian_features=[0, 1, 2], var_prior_weight=0.0).fit(X, y)
    ref = GaussianNB().fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-10)


def test_mixednb_prior_weight_shrinks_singleton_variances():
    X = np.array([[0.0], [0.0001], [5.0], [6.0], [7.0]])
    y = np.array([0, 1, 1, 2, 2])
    raw = MixedNB(gaussian_features=[0]).fit(X, y).estimators_["gaussian"].var_
    shrunk = (
        MixedNB(gaussian_features=[0], var_prior_weight=1.0)
        .fit(X, y)
        .estimators_["gaussian"]
        .var_
    )
    assert shrunk[0, 0] > 100 * raw[0, 0]
    pooled = X.var()
    assert np.all(shrunk <= pooled + 1e-6)


# --- Parent discretization --------------------------------------------------


def test_declared_gaussian_integer_parent_is_binned():
    # Integer-valued continuous attribute (e.g. an amount) with many distinct
    # values must be binned as a super-parent, not used as a categorical one.
    rng = np.random.RandomState(0)
    amount = rng.randint(100, 20000, size=400).astype(float)
    other = rng.normal(size=400)
    X = np.c_[amount, other]
    y = (other + rng.normal(scale=1.0, size=400) > 0).astype(int)
    model = AnDE(n_dependence=1, n_bins=5, gaussian_features=[0, 1]).fit(X, y)
    assert model._parent_cards[0] <= 5
    train_acc = np.mean(model.predict(X) == y)
    assert train_acc < 0.95  # no memorisation through a high-cardinality parent


def test_unseen_categorical_parent_value_deactivates_component():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 10, dtype=float)
    y = np.array([0, 1, 1, 0] * 10)
    model = AnDE(n_dependence=1, categorical_features=[0, 1]).fit(X, y)
    _, _, active, _ = model._get_jll_per_model(np.array([[7.0, 0.0]]))
    assert not active[0, 0]  # component with super-parent X0 = 7 (unseen)
    assert active[0, 1]


def test_laplace_unseen_state_uses_uniform_child_conditionals():
    # Parent X0=1 never co-occurs with class 1; child X1 has 3 categories.
    X = np.array([[0, 0], [0, 1], [1, 2], [1, 0], [0, 2], [0, 2], [0, 1]], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    kw = dict(n_dependence=1, categorical_features=[0, 1])
    lap = AnDE(unseen_state="laplace", **kw).fit(X, y)
    jll, _, active, _ = lap._get_jll_per_model(np.array([[1.0, 2.0]]))
    comp = lap.ensemble_[0]  # super-parent X0
    expected = comp["log_prior_unseen"] + np.log(1.0 / 3.0)
    np.testing.assert_allclose(jll[0, 1, 0], expected)
    back = AnDE(unseen_state="backoff", **kw).fit(X, y)
    jll_b, _, _, _ = back._get_jll_per_model(np.array([[1.0, 2.0]]))
    assert not np.isclose(jll_b[0, 1, 0], expected)


def test_invalid_unseen_state_raises():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="unseen_state"):
        AnDE(unseen_state="other", categorical_features=[0, 1]).fit(X, y)
