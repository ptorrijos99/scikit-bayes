"""Tests for the Tree Augmented Naive Bayes classifier: structure learning
(conditional mutual information and Chow-Liu tree) and parameter estimation,
checked against hand-computed values and a reference implementation."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pickle
from collections import Counter
from itertools import product

import numpy as np
import pytest
from scipy.stats import norm

from skbn import TAN, MixedNB

# --- Reference implementation (plain Python, discrete data) -----------------


def _reference_cmi(X, y):
    """I(A_i; A_j | C) in nats by explicit counting."""
    n, n_features = X.shape
    n_c = Counter(y.tolist())
    cmi = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
            n_cij = Counter(zip(y.tolist(), X[:, i].tolist(), X[:, j].tolist()))
            n_ci = Counter(zip(y.tolist(), X[:, i].tolist()))
            n_cj = Counter(zip(y.tolist(), X[:, j].tolist()))
            total = 0.0
            for (c, a, b), count in n_cij.items():
                total += (count / n) * np.log(
                    count * n_c[c] / (n_ci[(c, a)] * n_cj[(c, b)])
                )
            cmi[i, j] = total
    return cmi


def _spanning_trees(n_nodes):
    """All labelled spanning trees of K_n, as edge sets (Pruefer decoding)."""
    for seq in product(range(n_nodes), repeat=n_nodes - 2):
        degree = [1] * n_nodes
        for v in seq:
            degree[v] += 1
        edges = set()
        for v in seq:
            leaf = min(u for u in range(n_nodes) if degree[u] == 1)
            edges.add(frozenset((leaf, v)))
            degree[leaf] -= 1
            degree[v] -= 1
        u, w = [u for u in range(n_nodes) if degree[u] == 1]
        edges.add(frozenset((u, w)))
        yield edges


def _reference_proba(X_train, y_train, X_test, parents, alpha=1.0):
    """TAN posterior with Laplace-smoothed CPTs, by explicit counting.

    Uses an MLE class prior, as MixedNB does, and assumes integer categorical
    attributes with values in ``[0, max]``.
    """
    classes = np.unique(y_train)
    n = len(y_train)
    cards = X_train.max(axis=0) + 1
    out = np.zeros((len(X_test), len(classes)))
    for s, x in enumerate(X_test):
        for k, c in enumerate(classes):
            in_c = y_train == c
            p = in_c.sum() / n
            for i, pa in enumerate(parents):
                rows = in_c if pa < 0 else in_c & (X_train[:, pa] == x[pa])
                n_match = np.sum(rows & (X_train[:, i] == x[i]))
                p *= (n_match + alpha) / (rows.sum() + alpha * cards[i])
            out[s, k] = p
    return out / out.sum(axis=1, keepdims=True)


def _undirected_edges(parents):
    return {frozenset((i, int(p))) for i, p in enumerate(parents) if p >= 0}


@pytest.fixture
def categorical_data():
    """Five categorical attributes with a few strong pairwise dependencies."""
    rng = np.random.RandomState(0)
    n = 400
    y = rng.randint(0, 3, size=n)
    a0 = rng.randint(0, 4, size=n)
    a1 = np.where(rng.rand(n) < 0.8, (a0 + y) % 4, rng.randint(0, 4, size=n))
    a2 = rng.randint(0, 3, size=n)
    a3 = np.where(rng.rand(n) < 0.6, (a2 + y) % 3, rng.randint(0, 3, size=n))
    a4 = np.where(rng.rand(n) < 0.5, a1 % 3, rng.randint(0, 3, size=n))
    return np.column_stack([a0, a1, a2, a3, a4]), y


# --- Structure learning -----------------------------------------------------


def test_cmi_and_structure_by_hand():
    # A1 is a copy of A0; A2 is independent of A0 given C=0 only.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a0 = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    a2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X = np.column_stack([a0, a0, a2])
    model = TAN().fit(X, y)

    # I(A0; A1 | C) = H(A0 | C) = 1/2 H(1/2, 1/2) + 1/2 H(3/4, 1/4)
    h_c1 = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    i01 = 0.5 * np.log(2) + 0.5 * h_c1
    # Given C=1 the joint of (A0, A2) is {(0,0): 1/2, (0,1): 1/4, (1,1): 1/4}
    i02 = 0.5 * (
        0.5 * np.log(0.5 / 0.375) + 0.25 * np.log(0.25 / 0.375) + 0.25 * np.log(2)
    )
    expected = np.array([[0, i01, i02], [i01, 0, i02], [i02, i02, 0]])
    np.testing.assert_allclose(model.conditional_mutual_info_, expected, atol=1e-12)
    # A1 hangs from A0; A2 ties between A0 and A1 and keeps the first one
    np.testing.assert_array_equal(model.parents_, [-1, 0, 0])


def test_cmi_matches_reference(categorical_data):
    X, y = categorical_data
    model = TAN().fit(X, y)
    np.testing.assert_allclose(
        model.conditional_mutual_info_, _reference_cmi(X, y), atol=1e-12
    )


def test_tree_has_maximum_weight(categorical_data):
    X, y = categorical_data
    model = TAN().fit(X, y)
    w = model.conditional_mutual_info_

    def weight(edges):
        return sum(w[tuple(e)] for e in edges)

    best = max(_spanning_trees(X.shape[1]), key=weight)
    learned = _undirected_edges(model.parents_)
    assert len(learned) == X.shape[1] - 1
    assert learned == best


@pytest.mark.parametrize("root", [0, 2, 4])
def test_root_only_orients_the_tree(categorical_data, root):
    X, y = categorical_data
    reference = TAN(root=0).fit(X, y)
    model = TAN(root=root).fit(X, y)
    assert model.parents_[root] == -1
    assert np.sum(model.parents_ == -1) == 1
    assert _undirected_edges(model.parents_) == _undirected_edges(reference.parents_)
    # Every attribute reaches the root by following its parents
    for i in range(X.shape[1]):
        node, steps = i, 0
        while model.parents_[node] >= 0 and steps <= X.shape[1]:
            node, steps = model.parents_[node], steps + 1
        assert node == root


def test_recovers_chain_structure():
    # A0 -> A1 -> A2 -> A3 (noisy copies), all also dependent on the class
    rng = np.random.RandomState(1)
    n = 2000
    y = rng.randint(0, 2, size=n)
    cols = [(rng.randint(0, 4, size=n) + y) % 4]
    for _ in range(3):
        noise = rng.randint(0, 4, size=n)
        cols.append(np.where(rng.rand(n) < 0.85, cols[-1], noise))
    X = np.column_stack(cols)
    model = TAN().fit(X, y)
    np.testing.assert_array_equal(model.parents_, [-1, 0, 1, 2])


# --- Parameter estimation ---------------------------------------------------


def test_posterior_by_hand():
    X = np.array([[0, 0], [0, 0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = TAN().fit(X, y)
    np.testing.assert_array_equal(model.parents_, [-1, 0])

    # P(y) P(x0 | y) P(x1 | y, x0), Laplace smoothing with alpha = 1
    j00 = 1 / 2 * (3 / 5) * (3 / 4)  # x = [0, 0], y = 0
    j01 = 1 / 2 * (2 / 5) * (1 / 3)  # x = [0, 0], y = 1
    j10 = 1 / 2 * (2 / 5) * (2 / 3)  # x = [1, 1], y = 0
    j11 = 1 / 2 * (3 / 5) * (1 / 4)  # x = [1, 1], y = 1
    expected = np.array([[j00, j01], [j10, j11]])
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        model.predict_proba([[0, 0], [1, 1]]), expected, atol=1e-12
    )


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_posterior_matches_reference(categorical_data, alpha):
    X, y = categorical_data
    X_train, y_train, X_test = X[:300], y[:300], X[300:]
    model = TAN(alpha=alpha).fit(X_train, y_train)
    expected = _reference_proba(X_train, y_train, X_test, model.parents_, alpha)
    np.testing.assert_allclose(model.predict_proba(X_test), expected, atol=1e-10)


def test_single_feature_is_naive_bayes():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 1))
    y = (X[:, 0] + rng.normal(scale=0.5, size=100) > 0).astype(int)
    tan = TAN().fit(X, y)
    nb = MixedNB(var_prior_weight=1.0).fit(X, y)
    assert tan.cpts_ == {}
    np.testing.assert_allclose(tan.predict_proba(X), nb.predict_proba(X), atol=1e-12)


def test_gaussian_child_estimates():
    # Categorical parent (3 values), Gaussian child
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, size=n)
    x0 = rng.randint(0, 3, size=n)
    x1 = x0 * (2 * y - 1) + rng.normal(scale=0.3, size=n)
    X = np.column_stack([x0, x1])
    model = TAN(var_prior_weight=2.0).fit(X, y)
    np.testing.assert_array_equal(model.parents_, [-1, 0])

    gnb = model.cpts_[1].estimators_["gaussian"]
    eps = 1e-9 * x1.var()
    for k, key in enumerate(model.cpts_[1].classes_):
        rows = (y == key // 3) & (x0 == key % 3)
        n_k = rows.sum()
        var = (n_k * (x1[rows].var() + eps) + 2.0 * (x1.var() + eps)) / (n_k + 2.0)
        np.testing.assert_allclose(gnb.theta_[k, 0], x1[rows].mean())
        np.testing.assert_allclose(gnb.var_[k, 0], var)


def test_xor_beats_naive_bayes():
    rng = np.random.RandomState(42)
    X = rng.randn(400, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    assert TAN().fit(X, y).score(X, y) > 0.85
    assert MixedNB().fit(X, y).score(X, y) < 0.65


# --- Unseen states ----------------------------------------------------------


@pytest.fixture
def unseen_data():
    """Parent value 2 never occurs with class 1."""
    X = np.array(
        [[0, 0], [1, 1], [2, 2], [2, 1], [0, 1], [1, 2], [0, 0], [1, 0], [1, 1]]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


def test_unseen_state_is_uniform(unseen_data):
    X, y = unseen_data
    model = TAN().fit(X, y)
    X_test = np.array([[2, 0], [2, 2]])
    expected = _reference_proba(X, y, X_test, model.parents_)
    np.testing.assert_allclose(model.predict_proba(X_test), expected, atol=1e-12)


def test_unseen_parent_category_is_uninformed(unseen_data):
    X, y = unseen_data
    X = X[:, ::-1]  # child first, so the parent (root 1) is the second column
    model = TAN(root=1).fit(X, y)
    np.testing.assert_array_equal(model.parents_, [1, -1])
    X_test = np.array([[1, 7]])  # parent value 7 never seen
    jll = model._joint_log_likelihood(X_test)
    root_ll = model.backoff_._feature_log_likelihood(X_test)[:, :, 1]
    expected = model.backoff_.class_log_prior_ + root_ll - np.log(3)
    np.testing.assert_allclose(jll, expected, atol=1e-12)
    assert np.all(np.isfinite(model.predict_proba(X_test)))


def test_unseen_state_gaussian_child_is_pooled():
    rng = np.random.RandomState(0)
    x0 = np.array([0, 1, 2] * 20 + [0, 1] * 30)
    y = np.array([0] * 60 + [1] * 60)
    x1 = rng.normal(size=120) + y
    X = np.column_stack([x0, x1])
    model = TAN().fit(X, y)
    X_test = np.array([[2, 0.3]])  # (y=1, x0=2) was never observed
    jll = model._joint_log_likelihood(X_test)
    nb_ll = model.backoff_._feature_log_likelihood(X_test)
    base = model.backoff_.class_log_prior_ + nb_ll[:, :, 0]
    pooled = norm.logpdf(0.3, x1.mean(), np.sqrt(x1.var() + 1e-9 * x1.var()))
    np.testing.assert_allclose(jll[0, 1], base[0, 1] + pooled, atol=1e-10)
    # Class 0 did observe x0=2, so its own Gaussian is used
    gnb = model.cpts_[1].estimators_["gaussian"]
    k = list(model.cpts_[1].classes_).index(2)
    own = norm.logpdf(0.3, gnb.theta_[k, 0], np.sqrt(gnb.var_[k, 0]))
    np.testing.assert_allclose(jll[0, 0], base[0, 0] + own, atol=1e-10)


# --- Feature types, parameters and API --------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "quantile", "kmeans"])
def test_mixed_types(strategy):
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, size=n)
    X = np.column_stack(
        [
            rng.normal(size=n) + y,  # Gaussian
            rng.randint(0, 50, size=n).astype(float),  # integer-valued Gaussian
            rng.randint(0, 4, size=n),  # categorical
            rng.randint(0, 2, size=n),  # Bernoulli
            np.ones(n),  # constant
        ]
    )
    model = TAN(
        strategy=strategy,
        n_bins=4,
        gaussian_features=[0, 1],
        categorical_features=[2],
        bernoulli_features=[3],
    ).fit(X, y)
    types = model.backoff_.feature_types_
    assert sorted(types["gaussian"]) == [0, 1] and types["categorical"] == [2]
    # The declared integer-valued Gaussian is binned when it acts as a parent
    assert model._parent_data[:, 1].max() < 4
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2) and np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("root", [-1, 3, 1.5, True, "0"])
def test_invalid_root(root):
    X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="root must be an integer"):
        TAN(root=root).fit(X, y)


def test_string_labels_and_log_proba(categorical_data):
    X, y = categorical_data
    labels = np.array(["a", "b", "c"])[y]
    model = TAN().fit(X, labels)
    np.testing.assert_array_equal(model.classes_, ["a", "b", "c"])
    np.testing.assert_allclose(
        np.exp(model.predict_log_proba(X)), model.predict_proba(X)
    )
    assert set(model.predict(X)) <= {"a", "b", "c"}


def test_pickle_roundtrip(categorical_data):
    X, y = categorical_data
    model = TAN().fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.parents_, model.parents_)
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
