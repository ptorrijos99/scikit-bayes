"""Tests for the estimation details of the AnDE family: aggregation rules,
weight layouts, zero-count handling, variance shrinkage and analytic gradients."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.special import logsumexp
from sklearn.base import clone

from skbn import AnDE, GaussianALR, GaussianAnJE, MixedNB, WeightedAnDE

MEANS = ["arithmetic", "geometric"]
SCOPES = ["component", "factor"]
LEVELS = [1, 2, 3, 4]


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


def _training_design(model, X, y):
    design, _ = model._design(X)
    return design, np.eye(len(model.classes_))[np.searchsorted(model.classes_, y)]


# --- Aggregation rules ------------------------------------------------------


def test_geometric_ande_is_geometric_mean(mixed_data):
    X, y = mixed_data
    model = AnDE(n_dependence=1, mean="geometric", **_fit_kwargs()).fit(X, y)
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


@pytest.mark.parametrize("n_dependence", [0, 1, 2])
def test_factors_sum_to_the_component_joint(mixed_data, n_dependence):
    X, y = mixed_data
    model = AnDE(n_dependence=n_dependence, **_fit_kwargs()).fit(X, y)
    X_new = np.vstack([X[:40], [[0.1, -0.2, 0.3, 3.0, 0.0]]])
    jll = model._get_jll_per_model(X_new)[0]
    factors = model._get_jll_per_model(X_new, per_factor=True)[0]
    assert factors.shape == jll.shape + (1 + X.shape[1] - n_dependence,)
    np.testing.assert_allclose(factors.sum(axis=3), jll, atol=1e-8)


# --- Weight layouts ---------------------------------------------------------


@pytest.mark.parametrize("mean", MEANS)
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("level", LEVELS)
def test_generative_weights_reproduce_ande(mixed_data, mean, scope, level):
    X, y = mixed_data
    ande = AnDE(n_dependence=1, mean=mean, **_fit_kwargs()).fit(X, y)
    model = WeightedAnDE(
        n_dependence=1,
        mean=mean,
        weight_scope=scope,
        weight_level=level,
        max_iter=5,
        **_fit_kwargs(),
    ).fit(X, y)
    model.coef_theta_ = (
        np.zeros(model.n_weights_) if model._offset else np.ones(model.n_weights_)
    )
    np.testing.assert_allclose(
        model.predict_proba(X), ande.predict_proba(X), atol=1e-10
    )


def test_component_weight_counts(mixed_data):
    X, y = mixed_data
    n_models, n_classes = 5, 3
    combos = sum(
        info["n_parent_combos"]
        for info in AnDE(n_dependence=1, **_fit_kwargs()).fit(X, y).ensemble_
    )
    expected = {1: n_models, 2: combos, 3: n_models * n_classes, 4: combos * n_classes}
    for level, count in expected.items():
        model = WeightedAnDE(weight_level=level, max_iter=1, **_fit_kwargs())
        # Every parent value occurs in training; +1 for the unseen-key weight
        assert model.fit(X, y).n_weights_ == count + 1


def test_factor_weight_counts(mixed_data):
    # Factor weights of Zaidi et al. (2017), Sec. 5.4, inside each SPODE: the
    # root P(y, x_p) is valued by x_p; the two categorical children have 4
    # values and the three Gaussian ones a single (undiscretized) value. Only
    # the combinations seen in training get a weight, plus one for the rest.
    X, y = mixed_data
    model = WeightedAnDE(weight_scope="factor", max_iter=1, **_fit_kwargs())
    counts = {}
    for level in LEVELS:
        counts[level] = model.set_params(weight_level=level).fit(X, y).n_weights_
    combos = np.array([info["n_parent_combos"] for info in model.ensemble_])
    children = [info["child_indices"] for info in model.ensemble_]
    values = [np.array([4 if i >= 3 else 1 for i in ch]) for ch in children]
    assert counts[1] == len(combos) * X.shape[1] + 1
    assert counts[2] == sum(p + v.sum() for p, v in zip(combos, values)) + 1
    assert counts[3] == 3 * (counts[2] - 1) + 1

    # Level 4: the (parent value, child value) pairs that occur, times 3 classes
    codes = model._encoder.transform(X)[0]
    expected = 1
    for info in model.ensemble_:
        pk = codes[:, info["parent_indices"]] @ info["parent_strides"]
        expected += 3 * len(np.unique(pk))
        for i in info["child_indices"]:
            child = X[:, i].astype(int) if i >= 3 else np.zeros(len(X), dtype=int)
            expected += 3 * len(np.unique(np.column_stack([pk, child]), axis=0))
    assert counts[4] == expected


def test_factor_weights_are_bounded_by_the_data():
    # A categorical child with huge codes must not inflate the number of
    # weights, and its unseen values keep the generative (unit) weight.
    rng = np.random.RandomState(0)
    n = 200
    X = np.column_stack([rng.normal(size=n), rng.choice([0, 7, 99999], n)])
    y = (X[:, 0] + (X[:, 1] == 7) + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
    kw = dict(gaussian_features=[0], categorical_features=[1], max_iter=20)
    model = WeightedAnDE(weight_scope="factor", weight_level=4, **kw).fit(X, y)
    assert model.n_weights_ <= n * 2 * 2 * 2 + 1  # samples x models x terms x classes
    assert model.learned_weights_[-1] == 1.0
    X_new = np.array([[0.3, 12345.0]])
    proba = model.predict_proba(X_new)
    assert np.all(np.isfinite(proba)) and np.isclose(proba.sum(), 1.0)


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(weight_scope="attribute"), "weight_scope"),
        (dict(weight_level=5), "weight_level"),
        (dict(weight_level=0, weight_scope="factor"), "weight_level"),
        (dict(average="median"), "average"),
    ],
)
def test_invalid_weight_parameters_raise(mixed_data, params, match):
    X, y = mixed_data
    with pytest.raises(ValueError, match=match):
        WeightedAnDE(**params).fit(X, y)


# --- Modular optimization ---------------------------------------------------


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


@pytest.mark.parametrize(
    "scope, levels, mean",
    [
        ("component", [3, 4], "arithmetic"),
        ("component", LEVELS, "geometric"),
        ("factor", LEVELS, "arithmetic"),
    ],
)
def test_modular_identifiable_levels_learn(mixed_data, scope, levels, mean):
    X, y = mixed_data
    for level in levels:
        wande = WeightedAnDE(
            n_dependence=1,
            mean=mean,
            weight_scope=scope,
            weight_level=level,
            modular=True,
            **_fit_kwargs(),
        ).fit(X, y)
        assert not np.allclose(wande.learned_weights_, 1.0)


@pytest.mark.parametrize("mean", MEANS)
def test_modular_factor_weights_fit_each_component_alone(mixed_data, mean):
    # Modular factor weights make each SPODE a wCCBN fitted on its own CLL, so
    # they do not depend on the pooling of the components.
    X, y = mixed_data
    kw = dict(weight_scope="factor", weight_level=3, modular=True, **_fit_kwargs())
    model = WeightedAnDE(mean=mean, **kw).fit(X, y)
    reference = WeightedAnDE(mean="arithmetic", **kw).fit(X, y)
    np.testing.assert_allclose(
        model.learned_weights_, reference.learned_weights_, atol=1e-8
    )


# --- Average of posteriors --------------------------------------------------


def _component_log_posteriors(model, X, theta):
    """log P^w_m(y | x) of every component, shape (n, C, M)."""
    design, _ = model._design(X)
    n_samples, n_models = design.active.shape
    u = (design.operator @ theta).reshape(n_samples, -1, n_models)
    if design.intercept is not None:
        u = u + design.intercept
    return u - logsumexp(u, axis=1, keepdims=True), design.active


@pytest.mark.parametrize("scope", SCOPES)
def test_posterior_average_with_generative_weights_is_mean_of_posteriors(
    mixed_data, scope
):
    X, y = mixed_data
    model = WeightedAnDE(
        average="posterior", weight_scope=scope, weight_level=3, **_fit_kwargs()
    ).fit(X, y)
    model.coef_theta_ = (
        np.zeros(model.n_weights_) if model._offset else np.ones(model.n_weights_)
    )
    log_post, active = _component_log_posteriors(model, X, model.coef_theta_)
    assert active.all()
    expected = np.exp(log_post).mean(axis=2)
    np.testing.assert_allclose(model.predict_proba(X), expected, atol=1e-10)


def test_posterior_average_is_irrelevant_for_the_geometric_mean(mixed_data):
    X, y = mixed_data
    kw = dict(mean="geometric", weight_scope="factor", weight_level=3, max_iter=20)
    joint = WeightedAnDE(**kw, **_fit_kwargs()).fit(X, y)
    posterior = WeightedAnDE(average="posterior", **kw, **_fit_kwargs()).fit(X, y)
    np.testing.assert_allclose(
        posterior.predict_proba(X), joint.predict_proba(X), atol=1e-10
    )


def test_fedande_wande_averages_the_posteriors_of_per_component_wccbns(mixed_data):
    # AnDE^w of FedAnDE: a wCCBN per SPODE (factor weights at level 4), each
    # fitted on its own CLL, whose posteriors are averaged.
    X, y = mixed_data
    model = WeightedAnDE(
        average="posterior",
        weight_scope="factor",
        weight_level=4,
        modular=True,
        **_fit_kwargs(),
    ).fit(X, y)
    log_post, active = _component_log_posteriors(model, X, model.coef_theta_)
    expected = (np.exp(log_post) * active[:, None, :]).sum(axis=2)
    expected /= active.sum(axis=1)[:, None]
    np.testing.assert_allclose(model.predict_proba(X), expected, atol=1e-10)

    # The modular weights do not depend on how the components are pooled
    joint = model.set_params(average="joint").fit(X, y)
    np.testing.assert_allclose(joint.coef_theta_, model.coef_theta_)


def test_samples_without_active_component_are_left_out_of_the_fit(mixed_data):
    X, y = mixed_data
    kw = dict(weight_scope="factor", weight_level=3, min_parent_count=70)
    model = WeightedAnDE(**kw, **_fit_kwargs()).fit(X, y)
    design, Y = _training_design(model, X, y)
    kept = design.active.any(axis=1)
    assert not kept.all()
    Y = Y * kept[:, None]
    # They add nothing to the objective, whose gradient stays exact
    theta = model.coef_theta_ + np.random.RandomState(3).uniform(
        -0.2, 0.2, size=model.n_weights_
    )
    nll, grad = model._nll_and_grad(theta, design, Y)
    scores, _ = model._forward(theta, design)
    log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
    np.testing.assert_allclose(nll, -np.sum(log_prob[kept] * Y[kept]))
    num = approx_fprime(theta, lambda t: model._nll_and_grad(t, design, Y)[0], 1e-6)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-3)
    assert np.all(np.isfinite(model.predict_proba(X)))


# --- Analytic gradients -----------------------------------------------------


@pytest.mark.parametrize("mean", MEANS)
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("level", LEVELS)
def test_joint_gradient_matches_finite_differences(mixed_data, mean, scope, level):
    X, y = mixed_data
    model = WeightedAnDE(
        n_dependence=1,
        mean=mean,
        weight_scope=scope,
        weight_level=level,
        max_iter=1,
        **_fit_kwargs(),
    ).fit(X, y)
    design, Y = _training_design(model, X, y)
    rng = np.random.RandomState(1)
    theta = model.coef_theta_ + rng.uniform(-0.3, 0.3, size=model.n_weights_)
    _, grad = model._nll_and_grad(theta, design, Y)
    num = approx_fprime(theta, lambda t: model._nll_and_grad(t, design, Y)[0], 1e-6)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("level", [3, 4])
def test_posterior_average_gradient_matches_finite_differences(
    mixed_data, scope, level
):
    X, y = mixed_data
    model = WeightedAnDE(
        average="posterior",
        weight_scope=scope,
        weight_level=level,
        max_iter=1,
        **_fit_kwargs(),
    ).fit(X, y)
    design, Y = _training_design(model, X, y)
    rng = np.random.RandomState(1)
    theta = model.coef_theta_ + rng.uniform(-0.3, 0.3, size=model.n_weights_)
    _, grad = model._nll_and_grad(theta, design, Y)
    num = approx_fprime(theta, lambda t: model._nll_and_grad(t, design, Y)[0], 1e-6)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("mean", MEANS)
def test_component_gradient_matches_finite_differences(mixed_data, mean):
    X, y = mixed_data
    model = WeightedAnDE(
        mean=mean, weight_scope="factor", weight_level=4, max_iter=1, **_fit_kwargs()
    ).fit(X, y)
    design, Y = _training_design(model, X, y)
    lo, hi = model._component_bounds[1], model._component_bounds[2]
    rows = design.active[:, 1]
    sub = design.component(1, rows, lo, hi)
    theta = np.random.RandomState(2).uniform(0.7, 1.3, size=hi - lo)
    _, grad = model._nll_and_grad(theta, sub, Y[rows])
    num = approx_fprime(theta, lambda t: model._nll_and_grad(t, sub, Y[rows])[0], 1e-6)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-3)


# --- Zero counts ------------------------------------------------------------


def test_unseen_augmented_state_backs_off_instead_of_vetoing():
    # Parent value 1 never co-occurs with class 1 in training.
    X = np.array([[0, 0.1], [0, 0.2], [1, 0.3], [1, 0.4], [0, 2.1], [0, 2.2], [0, 2.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    model = AnDE(
        n_dependence=1,
        mean="geometric",
        categorical_features=[0],
        gaussian_features=[1],
        unseen_state="backoff",
    ).fit(X, y)
    jll, _, active, _ = model._get_jll_per_model(np.array([[1, 2.1]]))
    assert np.all(np.isfinite(jll))
    assert jll.min() > -100  # no hard -700 floor
    proba = model.predict_proba(np.array([[1, 2.1]]))
    assert proba[0, 1] > 0.5  # the continuous child still points to class 1
    # Laplace smoothing gives that state (component 0) the pooled Gaussian
    laplace = clone(model).set_params(unseen_state="laplace").fit(X, y)
    jll_laplace = laplace._get_jll_per_model(np.array([[1, 2.1]]))[0]
    assert not np.isclose(jll[0, 1, 0], jll_laplace[0, 1, 0])


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
    assert model._encoder.cards_[0] <= 5
    train_acc = np.mean(model.predict(X) == y)
    assert train_acc < 0.95  # no memorisation through a high-cardinality parent


def test_unseen_categorical_parent_value_deactivates_component():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 10, dtype=float)
    y = np.array([0, 1, 1, 0] * 10)
    model = AnDE(n_dependence=1, categorical_features=[0, 1]).fit(X, y)
    _, _, active, _ = model._get_jll_per_model(np.array([[7.0, 0.0]]))
    assert not active[0, 0]  # component with super-parent X0 = 7 (unseen)
    assert active[0, 1]


@pytest.mark.parametrize("mean", MEANS)
@pytest.mark.parametrize("unseen_state", ["laplace", "backoff"])
def test_unseen_value_of_a_constant_parent_drops_only_its_components(
    mean, unseen_state
):
    # Components whose parents differ only in constant attributes share their
    # scores; an unseen value of one constant attribute leaves out only the
    # components that have it as a parent
    rng = np.random.RandomState(1)
    X = np.column_stack(
        [rng.randint(0, 3, 200), np.zeros(200), rng.randint(0, 4, 200), np.zeros(200)]
    )
    y = (X[:, 0] + X[:, 2] + rng.randint(0, 2, 200)) % 3
    X_test = X[:40].copy()
    X_test[::2, 1] = 5.0
    X_test[::3, 3] = 6.0
    model = AnDE(
        n_dependence=2,
        mean=mean,
        unseen_state=unseen_state,
        categorical_features=[0, 1, 2, 3],
    ).fit(X, y)
    jll, _, active, _ = model._get_jll_per_model(X_test)
    assert (~active).any() and active.any(axis=1).all()
    if mean == "geometric":
        expected = np.where(active[:, None], jll, 0.0).sum(2) / active.sum(1)[:, None]
    else:
        expected = logsumexp(np.where(active[:, None], jll, -np.inf), axis=2)
    expected -= logsumexp(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(model.predict_log_proba(X_test), expected, atol=1e-10)


@pytest.mark.parametrize("Estimator", [AnDE, WeightedAnDE, GaussianAnJE, GaussianALR])
@pytest.mark.parametrize("n_dependence", [-1, 1.5, 3])
def test_invalid_n_dependence_raises(Estimator, n_dependence):
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="n_dependence"):
        Estimator(n_dependence=n_dependence).fit(X, y)
