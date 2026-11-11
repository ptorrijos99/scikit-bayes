"""Tests for the Tree Augmented Naive Bayes classifier: structure learning
(conditional mutual information and Chow-Liu tree) and parameter estimation,
checked against hand-computed values and a reference implementation."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

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
    np.testing.assert_array_equal(tan.parents_, [-1])
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

    cond = model._conditional(1)
    eps = 1e-9 * x1.var()
    assert len(cond["keys"]) == 6
    for k, key in enumerate(cond["keys"]):
        rows = (y == key // 3) & (x0 == key % 3)
        n_k = rows.sum()
        var = (n_k * (x1[rows].var() + eps) + 2.0 * (x1.var() + eps)) / (n_k + 2.0)
        assert cond["count"][k] == n_k
        np.testing.assert_allclose(cond["mean"][k], x1[rows].mean())
        np.testing.assert_allclose(cond["var"][k], var)


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
    # The child adds the same constant to every class: only the root counts
    root_nb = MixedNB().fit(X[:, [1]], y)
    expected = root_nb._joint_log_likelihood(X_test[:, [1]]) - np.log(3)
    np.testing.assert_allclose(jll, expected, atol=1e-12)
    np.testing.assert_allclose(
        model.predict_proba(X_test), root_nb.predict_proba(X_test[:, [1]])
    )


def test_unseen_state_gaussian_child_is_pooled():
    rng = np.random.RandomState(0)
    x0 = np.array([0, 1, 2] * 20 + [0, 1] * 30)
    y = np.array([0] * 60 + [1] * 60)
    x1 = rng.normal(size=120) + y
    X = np.column_stack([x0, x1])
    model = TAN().fit(X, y)
    X_test = np.array([[2, 0.3]])  # (y=1, x0=2) was never observed
    jll = model._joint_log_likelihood(X_test)
    base = MixedNB().fit(x0[:, None], y)._joint_log_likelihood(X_test[:, [0]])
    pooled = norm.logpdf(0.3, x1.mean(), np.sqrt(x1.var() + 1e-9 * x1.var()))
    np.testing.assert_allclose(jll[0, 1], base[0, 1] + pooled, atol=1e-10)
    # Class 0 did observe x0=2, so its own Gaussian is used
    cond = model._conditional(1)
    k = list(cond["keys"]).index(2)
    own = norm.logpdf(0.3, cond["mean"][k], np.sqrt(cond["var"][k]))
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
    assert model._encoder.cards_[1] <= 4
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2) and np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("root", [-1, 3, 1.5, True, "0"])
def test_invalid_root(root):
    X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="root"):
        TAN(root=root).fit(X, y)


# --- Compiled kernels ---------------------------------------------------------


def test_cmi_with_unequal_cardinalities_and_a_constant_attribute():
    rng = np.random.RandomState(3)
    n = 400
    y = rng.randint(0, 3, size=n)
    X = np.column_stack(
        [
            rng.randint(0, 2, size=n),
            (rng.randint(0, 7, size=n) + y) % 7,
            np.zeros(n, dtype=int),  # constant: exactly independent of the rest
            rng.randint(0, 4, size=n),
        ]
    )
    X[:, 3] = np.where(rng.rand(n) < 0.6, X[:, 1] % 4, X[:, 3])
    model = TAN(categorical_features=[0, 1, 2, 3]).fit(X, y)
    cmi = model.conditional_mutual_info_
    np.testing.assert_allclose(cmi, _reference_cmi(X, y), atol=1e-12)
    # Exact zeros keep the tie-breaking of the spanning tree deterministic
    assert np.all(cmi[2] == 0.0) and np.all(cmi[:, 2] == 0.0)
    assert model.parents_[2] == 0


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_n_jobs_gives_the_same_model(categorical_data, n_jobs):
    X, y = categorical_data
    rng = np.random.RandomState(0)
    X = np.column_stack([X, rng.normal(size=len(y)) + y])
    serial = TAN().fit(X, y)
    threaded = TAN(n_jobs=n_jobs).fit(X, y)
    np.testing.assert_array_equal(threaded.parents_, serial.parents_)
    np.testing.assert_array_equal(
        threaded.conditional_mutual_info_, serial.conditional_mutual_info_
    )
    np.testing.assert_allclose(threaded.predict_proba(X), serial.predict_proba(X))


def _high_cardinality_data(n, seed):
    """Attributes of hundreds of values next to small ones, coded with the
    values ``0..card - 1`` all observed. The (class, a, b) tables of the
    large pairs and the (state, value) tables of their children have many
    more cells than samples, so they are counted on the non-empty cells."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=n)
    a0 = rng.randint(0, 200, size=n)
    a1 = np.where(rng.rand(n) < 0.7, (3 * a0 + y) % 150, rng.randint(0, 150, size=n))
    a2 = rng.randint(0, 3, size=n)
    a3 = np.where(rng.rand(n) < 0.6, (a2 + y) % 4, rng.randint(0, 4, size=n))
    a4 = np.where(rng.rand(n) < 0.5, a1 % 5, rng.randint(0, 5, size=n))
    X = np.column_stack([a0, a1, a2, a3, a4])
    X = np.column_stack([np.unique(col, return_inverse=True)[1] for col in X.T])
    return X, y


def test_high_cardinality_matches_reference():
    X, y = _high_cardinality_data(300, 0)
    model = TAN(categorical_features=range(5)).fit(X, y)
    np.testing.assert_allclose(
        model.conditional_mutual_info_, _reference_cmi(X, y), atol=1e-12
    )
    # Both layouts are in use: some discrete children keep only their
    # non-empty cells, others the dense table
    cell_off, disc_off = model._tables.cell_off, model._tables.disc_off
    sparse = np.diff(cell_off) > 0
    assert sparse.any() and not sparse.all()
    assert np.all(np.diff(disc_off)[sparse] <= len(y))
    # Training rows and rows whose (parent, child) combinations were mostly
    # never seen together
    rng = np.random.RandomState(1)
    X_test = np.vstack([X, np.column_stack([rng.permutation(c) for c in X.T])])
    expected = _reference_proba(X, y, X_test, model.parents_)
    np.testing.assert_allclose(model.predict_proba(X_test), expected, atol=1e-10)
    for i in np.flatnonzero(model.parents_ >= 0):
        p = model.parents_[i]
        cond = model._conditional(i)
        n_values = X[:, i].max() + 1
        for key, log_prob in zip(cond["keys"], cond["log_prob"]):
            c, a = divmod(key, X[:, p].max() + 1)
            rows = (y == c) & (X[:, p] == a)
            counts = np.bincount(X[rows, i], minlength=n_values)
            np.testing.assert_allclose(
                log_prob, np.log((counts + 1.0) / (rows.sum() + n_values)), atol=1e-12
            )


def test_high_cardinality_memory_follows_the_samples():
    """Two attributes of 4000 values: a dense (class, a, b) table or child
    table would have 16-32 million cells (128-256 MB); the fit allocates
    a few MB. tracemalloc sees the NumPy arrays of the fit and, when the
    kernels run as Python (NUMBA_DISABLE_JIT=1), those of the kernels."""
    import tracemalloc

    rng = np.random.RandomState(0)
    n = 4000
    y = rng.randint(0, 2, size=n)
    a0 = rng.permutation(n)
    a1 = np.where(rng.rand(n) < 0.8, (a0 + y) % n, rng.randint(0, n, size=n))
    X = np.column_stack(
        [a0, np.unique(a1, return_inverse=True)[1], rng.randint(0, 3, size=n)]
    )
    TAN(categorical_features=range(3)).fit(X[:50], y[:50])
    tracemalloc.start()
    model = TAN(categorical_features=range(3)).fit(X, y)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert model.parents_.tolist() == [-1, 0, 0]
    assert peak < 20 * 2**20


def test_wide_data():
    """Many more attributes than samples, the shape of microarray data."""
    rng = np.random.RandomState(0)
    n, d = 40, 300
    y = rng.randint(0, 2, size=n)
    X = rng.normal(size=(n, d)) + 0.5 * y[:, None]
    model = TAN(n_bins=3).fit(X, y)
    assert np.sum(model.parents_ < 0) == 1
    assert len(np.unique(model.parents_)) > 1
    proba = model.predict_proba(X)
    assert np.all(np.isfinite(proba)) and model.score(X, y) > 0.9
