"""Tests for the n-join models (AnJE and ALR, and their versions on the
factors of the SPnDEs) against the formulas of Zaidi et al. (2016)."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import combinations
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import approx_fprime
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

from skbn import ALR, AnJE, GaussianALR, GaussianAnJE
from skbn import anje as anje_module
from skbn._kernels import join_rows


@pytest.fixture
def discrete_data():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 3, size=(300, 4))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.randint(0, 2, 300)) % 3
    return X, y


def _log_cond(X, y, cols, X_eval, m, n_classes=3, card=3):
    """log of the m-estimate P(x_cols | y) of Zaidi et al. (2016) at X_eval."""
    shape = [card] * len(cols)
    keys = np.ravel_multi_index(X[:, cols].T, shape)
    eval_keys = np.ravel_multi_index(X_eval[:, cols].T, shape)
    out = np.empty((len(X_eval), n_classes))
    for c in range(n_classes):
        counts = np.bincount(keys[y == c], minlength=card ** len(cols))
        out[:, c] = np.log(
            (counts[eval_keys] + m / card ** len(cols)) / ((y == c).sum() + m)
        )
    return out


def _paper_scores(X, y, X_eval, n, m, join_weight):
    """log P(y) + join_weight * sum_alpha log P(x_alpha | y)."""
    counts = np.bincount(y, minlength=3)
    prior = np.log((counts + m / 3) / (len(y) + m))
    joins = sum(
        _log_cond(X, y, list(alpha), X_eval, m)
        for alpha in combinations(range(X.shape[1]), n)
    )
    return prior + join_weight * joins


def _normalize(scores):
    return scores - logsumexp(scores, axis=1, keepdims=True)


# --- AnJE -------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("m", [1.0, 0.5])
def test_anje_matches_paper_formula(discrete_data, n, m):
    X, y = discrete_data
    model = AnJE(n_join=n, alpha=m).fit(X, y)
    expected = _paper_scores(X, y, X, n, m, 1.0 / comb(X.shape[1] - 1, n - 1))
    assert_allclose(model.predict_log_proba(X), _normalize(expected), atol=1e-10)


def test_anje_joins_are_all_subsets(discrete_data):
    X, y = discrete_data
    model = AnJE(n_join=2).fit(X, y)
    assert_array_equal(model.joins_, list(combinations(range(4), 2)))


def test_anje_with_more_join_size_than_features_uses_the_full_join(discrete_data):
    X, y = discrete_data
    full = AnJE(n_join=10).fit(X, y)
    assert full.joins_.shape == (1, 4)
    expected = AnJE(n_join=4).fit(X, y).predict_proba(X)
    assert_allclose(full.predict_proba(X), expected)


def test_anje_unseen_value_falls_in_an_empty_cell(discrete_data):
    X, y = discrete_data
    model = AnJE(n_join=2).fit(X, y)
    x_new = np.array([[7, 1, 2, 0]])  # value 7 of attribute 0 never seen
    counts = np.bincount(y, minlength=3)
    expected = np.log((counts + 1 / 3) / (len(y) + 1))
    for alpha in combinations(range(4), 2):
        if 0 in alpha:  # m-estimate of a cell with no counts, |X_alpha| = 9
            expected = expected + np.log((1 / 9) / (counts + 1)) / 3
        else:
            expected = expected + _log_cond(X, y, list(alpha), x_new, 1.0) / 3
    assert_allclose(model.predict_log_proba(x_new), _normalize(expected), atol=1e-10)


def test_anje_solves_categorical_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 3)
    y = np.array([0, 1, 1, 0] * 3)
    assert_array_equal(AnJE(n_join=2).fit(X, y).predict(X), y)


def test_anje_bins_continuous_attributes():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=200), rng.randint(0, 3, 200)])
    y = (X[:, 0] > 0).astype(int)
    model = AnJE(n_join=2, n_bins=4).fit(X, y)
    assert model._encoder.cards_[0] == 4
    assert (model.predict(X) == y).mean() > 0.9


@pytest.mark.parametrize("n_join", [0, -1, 1.5])
def test_anje_invalid_n_join_raises(discrete_data, n_join):
    X, y = discrete_data
    with pytest.raises(ValueError, match="n_join"):
        AnJE(n_join=n_join).fit(X, y)


# --- ALR --------------------------------------------------------------------


def test_alr_with_unit_weights_is_the_unaveraged_product(discrete_data):
    X, y = discrete_data
    model = ALR(n_join=2, max_iter=1).fit(X, y)
    model.learned_weights_ = np.ones_like(model.learned_weights_)
    expected = _paper_scores(X, y, X, 2, 1.0, 1.0)
    assert_allclose(model.predict_log_proba(X), _normalize(expected), atol=1e-10)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_alr_with_anje_weights_is_anje(discrete_data, level):
    # ALR (init) of Appendix B: weights equal to Eq. 10 recover the AnJE
    X, y = discrete_data
    model = ALR(n_join=2, weight_level=level, max_iter=1).fit(X, y)
    weights = np.full((model.n_weights_ // model._n_columns, model._n_columns), 1 / 3)
    weights[0] = 1.0  # class prior
    model.learned_weights_ = weights.ravel()
    assert_allclose(
        model.predict_proba(X), AnJE(n_join=2).fit(X, y).predict_proba(X), atol=1e-10
    )


def test_alr_weight_counts(discrete_data):
    X, y = discrete_data
    joins = list(combinations(range(4), 2))
    seen_cells = sum(len(np.unique(X[:, list(j)], axis=0)) for j in joins)
    rows = 1 + seen_cells + len(joins)  # prior, seen cells, one empty cell per join
    assert ALR(weight_level=1, max_iter=1).fit(X, y).n_weights_ == 1 + len(joins)
    assert ALR(weight_level=2, max_iter=1).fit(X, y).n_weights_ == rows
    assert ALR(weight_level=3, max_iter=1).fit(X, y).n_weights_ == rows * 3


@pytest.mark.parametrize("level", [1, 2, 3])
def test_alr_gradient_matches_finite_differences(discrete_data, level):
    X, y = discrete_data
    model = ALR(n_join=2, weight_level=level, max_iter=1).fit(X, y)
    codes_t = model._transform_codes(X)
    theta = np.random.RandomState(1).uniform(0.5, 1.5, size=model.n_weights_)
    _, grad = model._nll_and_grad(theta, codes_t, y)
    numeric = approx_fprime(
        theta, lambda t: model._nll_and_grad(t, codes_t, y)[0], 1e-6
    )
    assert_allclose(grad, numeric, rtol=1e-4, atol=1e-3)


def test_alr_learns_and_beats_anje_on_training_likelihood(discrete_data):
    X, y = discrete_data
    alr = ALR(n_join=2, max_iter=200).fit(X, y)
    anje = AnJE(n_join=2).fit(X, y)
    rows = np.arange(len(y))
    assert (
        alr.predict_log_proba(X)[rows, y].sum()
        > anje.predict_log_proba(X)[rows, y].sum()
    )


def test_alr_unseen_cells_keep_unit_weight(discrete_data):
    X, y = discrete_data
    model = ALR(n_join=2, weight_level=3, l2_reg=1e-2, max_iter=50).fit(X, y)
    # Rows of the cells seen in training first, then the empty row of every join
    seen = np.arange(model._n_rows) < len(model._cell_keys)
    weights = model.learned_weights_.reshape(-1, 3)
    assert (~seen).sum() == len(model.joins_)  # the empty cell of every join
    assert_allclose(weights[~seen], 1.0)
    assert not np.allclose(weights[seen], 1.0)


def test_alr_weights_are_bounded_by_the_data():
    # A high-cardinality attribute must not inflate the tables or the weights
    rng = np.random.RandomState(0)
    X = np.column_stack(
        [
            rng.randint(0, 100000, 200),
            rng.randint(0, 50000, 200),
            rng.randint(0, 3, 200),
        ]
    )
    y = rng.randint(0, 2, 200)
    model = ALR(n_join=2, categorical_features=[0, 1, 2], max_iter=5).fit(X, y)
    assert model._n_rows <= 1 + len(X) * len(model.joins_) + len(model.joins_)
    assert model.n_weights_ == model._n_rows * 2


def test_alr_regularization_pulls_weights_towards_one(discrete_data):
    X, y = discrete_data
    loose = ALR(l2_reg=1e-6, max_iter=200).fit(X, y)
    tight = ALR(l2_reg=10.0, max_iter=200).fit(X, y)
    assert (
        np.abs(tight.learned_weights_ - 1).mean()
        < np.abs(loose.learned_weights_ - 1).mean()
    )


def test_alr_invalid_parameters_raise(discrete_data):
    X, y = discrete_data
    with pytest.raises(ValueError, match="weight_level"):
        ALR(weight_level=4).fit(X, y)
    with pytest.raises(ValueError, match="alpha"):
        ALR(alpha=0.0).fit(X, y)


def test_alr_default_stopping_is_that_of_the_original_code():
    # factr = 1e7 of L-BFGS-B, i.e. a relative reduction of 1e7 epsilons
    assert ALR().tol == pytest.approx(1e7 * np.finfo(float).eps)
    assert GaussianALR().tol == ALR().tol


def test_binary_search_of_cells_matches_the_dense_map(monkeypatch):
    # Key spaces too large for the dense map fall back to a binary search;
    # attributes of different cardinalities and values unseen in training
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.randint(0, c, 300) for c in (2, 3, 4, 6)])
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.randint(0, 2, 300)) % 3
    X_new = np.vstack([X[:20], [[9, 0, 1, 2], [0, 7, 7, 1], [1, 2, 3, 8]]])
    dense = [AnJE(n_join=2).fit(X, y), ALR(n_join=2, max_iter=20).fit(X, y)]
    monkeypatch.setattr(anje_module, "_MAX_DENSE_KEYS", 0)
    for model in dense:
        sparse = clone(model).fit(X, y)
        assert sparse._row_of_cell.size == 0
        assert_allclose(sparse.predict_log_proba(X_new), model.predict_log_proba(X_new))


@pytest.mark.parametrize("cached", [True, False])
def test_two_class_alr_objective_matches_the_predictions(cached):
    # Two classes have their own loop in the objective, which scores four
    # samples at a time (301 samples leave a remainder)
    rng = np.random.RandomState(3)
    X = rng.randint(0, 3, size=(301, 5))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.randint(0, 2, 301)) % 2
    model = ALR(n_join=2, max_iter=1).fit(X, y)
    codes_t = model._transform_codes(X)
    rows = join_rows(codes_t, *model._cell_args(), True) if cached else None
    theta = np.random.RandomState(1).uniform(0.5, 1.5, size=model.n_weights_)
    nll, grad = model._nll_and_grad(theta, codes_t, y, rows)
    model.learned_weights_ = theta
    expected = -model.predict_log_proba(X)[np.arange(len(y)), y].sum()
    assert_allclose(nll, expected, rtol=1e-10)
    numeric = approx_fprime(
        theta, lambda t: model._nll_and_grad(t, codes_t, y, rows)[0], 1e-6
    )
    assert_allclose(grad, numeric, rtol=1e-4, atol=1e-3)


def test_alr_objective_without_cached_rows(discrete_data, monkeypatch):
    X, y = discrete_data
    cached = ALR(n_join=2, max_iter=20).fit(X, y)
    monkeypatch.setattr(anje_module, "_MAX_CACHED_ROWS", 0)
    computed = ALR(n_join=2, max_iter=20).fit(X, y)
    assert_allclose(computed.learned_weights_, cached.learned_weights_)


@pytest.mark.parametrize(
    "model", [AnJE(n_join=2), ALR(n_join=2, max_iter=20)], ids=repr
)
def test_threads_give_the_same_model(discrete_data, model):
    X, y = discrete_data
    single = clone(model).fit(X, y).predict_proba(X)
    threaded = clone(model).set_params(n_jobs=2).fit(X, y).predict_proba(X)
    assert_allclose(threaded, single, atol=1e-8)


# --- AnJE and ALR on the factors of the SPnDEs --------------------------------


def _spnde_join_scores(X, y, X_eval, n, alpha, card=3, n_classes=3):
    """log P(y) + mean_P [r log P(x_P | y) + c sum_j log P(x_j | y, x_P)], with
    the Laplace estimates of the SPnDEs computed from the counts."""
    d = X.shape[1]
    r, c = d / (n + 1), d / ((n + 1) * (d - n))
    parent_sets = [list(P) for P in combinations(range(d), n)]
    out = np.tile(
        np.log(np.bincount(y, minlength=n_classes) / len(y)), (len(X_eval), 1)
    )
    for P in parent_sets:
        for cls in range(n_classes):
            X_c = X[y == cls]
            same_parents = (X_c[:, P][None] == X_eval[:, P][:, None]).all(axis=2)
            n_parents = same_parents.sum(axis=1)
            term = r * np.log((n_parents + alpha) / (len(X_c) + alpha * card**n))
            for j in sorted(set(range(d)) - set(P)):
                same_child = X_c[:, j][None] == X_eval[:, j][:, None]
                n_child = (same_parents & same_child).sum(axis=1)
                term += c * np.log((n_child + alpha) / (n_parents + alpha * card))
            out[:, cls] += term / len(parent_sets)
    return out


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gaussian_anje_matches_the_laplace_join_formula(discrete_data, n):
    X, y = discrete_data
    model = GaussianAnJE(
        n_dependence=n, categorical_features=[0, 1, 2, 3], unseen_state="laplace"
    ).fit(X, y)
    expected = _spnde_join_scores(X, y, X, n, 1.0)
    assert_allclose(model.predict_log_proba(X), _normalize(expected), atol=1e-10)


@pytest.mark.parametrize("n", [1, 2])
def test_gaussian_anje_tends_to_anje_as_smoothing_vanishes(n):
    # With every cell observed, both reduce to the maximum-likelihood AnJE
    rng = np.random.RandomState(0)
    X = rng.randint(0, 3, size=(3000, 4))
    y = (X[:, 0] + rng.randint(0, 3, 3000)) % 3
    gaussian = GaussianAnJE(
        n_dependence=n, alpha=1e-9, categorical_features=[0, 1, 2, 3]
    ).fit(X, y)
    anje = AnJE(n_join=n + 1, alpha=1e-9).fit(X, y)
    assert_allclose(gaussian.predict_proba(X), anje.predict_proba(X), atol=1e-6)


def test_gaussian_anje_of_order_zero_is_gaussian_nb():
    X, y = load_iris(return_X_y=True)
    model = GaussianAnJE(n_dependence=0).fit(X, y)
    expected = GaussianNB().fit(X, y).predict_proba(X)
    assert_allclose(model.predict_proba(X), expected, atol=1e-12)


def test_gaussian_anje_is_not_biased_towards_rare_classes():
    # The smoothed augmented prior gives a rare class far more mass than
    # P(y); the root factor must be divided by it, or the AnJE exponent
    # d / (n+1) turns that excess into a bias towards the rare class.
    rng = np.random.RandomState(0)
    X = rng.randint(0, 8, size=(605, 20))
    y = np.repeat([0, 1, 2], [300, 300, 5])
    model = GaussianAnJE(n_dependence=1, categorical_features=list(range(20)))
    predicted = model.fit(X, y).predict(X)
    assert (predicted == 2).mean() < 0.05


def test_gaussian_anje_keeps_continuous_children_gaussian():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Two points in the same bin of every attribute
    pair = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    binned = AnJE(n_join=2).fit(X, y).predict_proba(pair)
    gaussian = GaussianAnJE(n_dependence=1).fit(X, y).predict_proba(pair)
    assert_allclose(binned[0], binned[1])
    assert not np.allclose(gaussian[0], gaussian[1])


@pytest.mark.parametrize("n", [0, 1, 2])
def test_gaussian_alr_with_unit_weights_is_gaussian_anje(n):
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=200), rng.randint(0, 3, size=(200, 3))])
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] == 2)
    kwargs = dict(n_dependence=n, categorical_features=[1, 2, 3])
    alr = GaussianALR(max_iter=1, **kwargs).fit(X, y)
    alr.coef_theta_ = np.ones_like(alr.coef_theta_)
    expected = GaussianAnJE(**kwargs).fit(X, y).predict_log_proba(X)
    assert_allclose(alr.predict_log_proba(X), expected, atol=1e-10)


def test_gaussian_alr_gradient_matches_finite_differences():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=60), rng.randint(0, 3, size=(60, 2))])
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] == 2)
    model = GaussianALR(categorical_features=[1, 2], max_iter=1).fit(X, y)
    design, _ = model._design(X)
    Y = np.eye(3)[y]
    theta = rng.uniform(0.5, 1.5, size=model.n_weights_)
    _, grad = model._nll_and_grad(theta, design, Y)
    numeric = approx_fprime(theta, lambda t: model._nll_and_grad(t, design, Y)[0], 1e-6)
    assert_allclose(grad, numeric, rtol=1e-4, atol=1e-3)


def test_gaussian_alr_beats_gaussian_anje_on_training_likelihood(discrete_data):
    X, y = discrete_data
    kwargs = dict(n_dependence=1, categorical_features=[0, 1, 2, 3])
    alr = GaussianALR(max_iter=200, **kwargs).fit(X, y)
    anje = GaussianAnJE(**kwargs).fit(X, y)
    rows = np.arange(len(y))
    assert (
        alr.predict_log_proba(X)[rows, y].sum()
        > anje.predict_log_proba(X)[rows, y].sum()
    )


def test_gaussian_alr_requires_positive_alpha(discrete_data):
    X, y = discrete_data
    with pytest.raises(ValueError, match="alpha"):
        GaussianALR(alpha=0.0).fit(X, y)
