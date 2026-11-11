"""Tests for the K-Dependence Bayesian classifier: structure learning (mutual
information, attribute order and parent selection) and parameter estimation,
checked against hand-computed values and a reference implementation."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter

import numpy as np
import pytest

from skbn import KDB, TAN, MixedNB

# --- Reference implementation (plain Python, discrete data) -----------------


def _entropy(*p):
    return -sum(q * np.log(q) for q in p if q > 0)


def _reference_mi(X, y):
    """I(A_i; C) in nats by explicit counting."""
    n, n_features = X.shape
    n_c = Counter(y.tolist())
    mi = np.zeros(n_features)
    for i in range(n_features):
        n_a = Counter(X[:, i].tolist())
        n_ca = Counter(zip(y.tolist(), X[:, i].tolist()))
        mi[i] = sum(
            (count / n) * np.log(count * n / (n_c[c] * n_a[a]))
            for (c, a), count in n_ca.items()
        )
    return mi


def _reference_structure(mi, cmi, k):
    """Sahami's algorithm step by step: repeatedly add the remaining attribute
    with the highest I(A_i; C), with arcs from the min(k, |S|) attributes of
    S with the highest I(A_i; A_j | C). Ties keep the lowest index and the
    attribute placed first."""
    remaining = list(range(len(mi)))
    placed, parents = [], {}
    while remaining:
        x_max = max(remaining, key=lambda i: (mi[i], -i))
        ranked = sorted(placed, key=lambda j: -cmi[x_max, j])
        parents[x_max] = ranked[: min(k, len(placed))]
        placed.append(x_max)
        remaining.remove(x_max)
    return placed, [parents[i] for i in range(len(mi))]


def _reference_proba(X_train, y_train, X_test, parents, alpha=1.0):
    """KDB posterior with Laplace-smoothed CPTs, by explicit counting.

    Uses an MLE class prior, as MixedNB does, and assumes integer categorical
    attributes with values in ``[0, max]``.
    """
    classes = np.unique(y_train)
    n = len(y_train)
    cards = X_train.max(axis=0) + 1
    out = np.zeros((len(X_test), len(classes)))
    for s, x in enumerate(X_test):
        for c_idx, c in enumerate(classes):
            in_c = y_train == c
            p = in_c.sum() / n
            for i, pa in enumerate(parents):
                rows = in_c & np.all(X_train[:, pa] == x[pa], axis=1)
                n_match = np.sum(rows & (X_train[:, i] == x[i]))
                p *= (n_match + alpha) / (rows.sum() + alpha * cards[i])
            out[s, c_idx] = p
    return out / out.sum(axis=1, keepdims=True)


def _as_lists(parents):
    return [p.tolist() for p in parents]


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


@pytest.fixture
def hand_data():
    # Column 2 is a copy of the class and column 3 a copy of column 1;
    # column 0 is uniform in both classes.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a0 = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    a2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return np.column_stack([a2, a0, y, a0]), y


def test_mutual_info_by_hand(hand_data):
    X, y = hand_data
    model = KDB().fit(X, y)

    # I(A; C) = H(A) - H(A | C); P(a0 = 1) = 3/8, P(a0 = 1 | c) = 1/2, 1/4
    m = _entropy(3 / 8, 5 / 8) - 0.5 * np.log(2) - 0.5 * _entropy(1 / 4, 3 / 4)
    np.testing.assert_allclose(model.mutual_info_, [0, m, np.log(2), m], atol=1e-12)
    # I(A0; A1 | C) = H(A0 | C) for a copy; the class copy is independent of
    # everything given the class
    i01 = 0.5 * np.log(2) + 0.5 * _entropy(1 / 4, 3 / 4)
    i02 = 0.5 * (
        0.5 * np.log(0.5 / 0.375) + 0.25 * np.log(0.25 / 0.375) + 0.25 * np.log(2)
    )
    expected = np.array(
        [[0, i02, 0, i02], [i02, 0, 0, i01], [0, 0, 0, 0], [i02, i01, 0, 0]]
    )
    np.testing.assert_allclose(model.conditional_mutual_info_, expected, atol=1e-12)


@pytest.mark.parametrize(
    "k, expected",
    [
        (0, [[], [], [], []]),
        # Column 1 can only take the class copy; column 3 prefers its copy;
        # column 0 ties between columns 1 and 3 and keeps the one placed first
        (1, [[1], [2], [], [1]]),
        (2, [[1, 3], [2], [], [1, 2]]),
        (3, [[1, 3, 2], [2], [], [1, 2]]),
    ],
)
def test_structure_by_hand(hand_data, k, expected):
    X, y = hand_data
    model = KDB(k=k).fit(X, y)
    # Decreasing I(A_i; C); columns 1 and 3 tie and keep the index order
    np.testing.assert_array_equal(model.feature_order_, [2, 1, 3, 0])
    assert _as_lists(model.parents_) == expected


def test_mutual_info_matches_reference(categorical_data):
    X, y = categorical_data
    model = KDB().fit(X, y)
    np.testing.assert_allclose(model.mutual_info_, _reference_mi(X, y), atol=1e-12)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 10])
def test_structure_matches_reference(categorical_data, k):
    X, y = categorical_data
    model = KDB(k=k).fit(X, y)
    order, parents = _reference_structure(
        model.mutual_info_, model.conditional_mutual_info_, k
    )
    np.testing.assert_array_equal(model.feature_order_, order)
    assert _as_lists(model.parents_) == parents
    # The t-th attribute placed takes min(k, t) of the attributes placed before
    for t, i in enumerate(model.feature_order_):
        assert len(model.parents_[i]) == min(k, t)
        assert set(model.parents_[i]) <= set(model.feature_order_[:t])
    n_parents, children = model._tables.n_parents, model._tables.children
    assert dict(zip(children.tolist(), n_parents.tolist())) == {
        i: len(p) for i, p in enumerate(model.parents_)
    }


def _chain_data(n, rng):
    """C -> A0 -> A1 -> A2 -> A3, each attribute a noisy copy of the previous
    one, so I(A_i; C) decreases along the chain."""
    y = rng.randint(0, 2, size=n)
    cols = [np.where(rng.rand(n) < 0.8, y, rng.randint(0, 4, size=n))]
    for _ in range(3):
        noise = rng.randint(0, 4, size=n)
        cols.append(np.where(rng.rand(n) < 0.85, cols[-1], noise))
    return np.column_stack(cols), y


def test_recovers_chain_and_matches_tan():
    X, y = _chain_data(2000, np.random.RandomState(1))
    model = KDB(k=1).fit(X, y)
    np.testing.assert_array_equal(model.feature_order_, [0, 1, 2, 3])
    assert _as_lists(model.parents_) == [[], [0], [1], [2]]
    # Same network as TAN rooted at the first attribute, hence same model
    tan = TAN(root=0).fit(X, y)
    np.testing.assert_array_equal(tan.parents_, [-1, 0, 1, 2])
    np.testing.assert_allclose(model.predict_proba(X), tan.predict_proba(X))
    # With two parents the closest attribute of the chain comes first
    model = KDB(k=2).fit(X, y)
    assert _as_lists(model.parents_) == [[], [0], [1, 0], [2, 1]]


def test_parity_needs_two_parents():
    # y = x0 xor x1 xor x2: no attribute, or pair of attributes, is informative
    rng = np.random.RandomState(0)

    def make(n):
        X = rng.randint(0, 2, size=(n, 3))
        return X, np.bitwise_xor.reduce(X, axis=1)

    X_train, y_train = make(500)
    X_test, y_test = make(1000)
    assert KDB(k=2).fit(X_train, y_train).score(X_test, y_test) == 1.0
    for k in (0, 1):
        assert KDB(k=k).fit(X_train, y_train).score(X_test, y_test) < 0.65


# --- Parameter estimation ---------------------------------------------------


def test_posterior_by_hand():
    X = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = KDB(k=2).fit(X, y)
    np.testing.assert_array_equal(model.feature_order_, [0, 1, 2])
    assert sorted(model.parents_[2].tolist()) == [0, 1]

    # P(y) P(x0 | y) P(x1 | y, x0) P(x2 | y, x0, x1), Laplace with alpha = 1;
    # (y=1, x0=0) never occurs, so its conditionals are uniform
    j00 = 1 / 2 * (4 / 6) * (2 / 5) * (1 / 3)  # x = [0, 1, 1], y = 0
    j01 = 1 / 2 * (1 / 6) * (1 / 2) * (1 / 2)  # x = [0, 1, 1], y = 1
    j10 = 1 / 2 * (2 / 6) * (2 / 3) * (2 / 3)  # x = [1, 1, 1], y = 0
    j11 = 1 / 2 * (5 / 6) * (4 / 6) * (3 / 5)  # x = [1, 1, 1], y = 1
    expected = np.array([[j00, j01], [j10, j11]])
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        model.predict_proba([[0, 1, 1], [1, 1, 1]]), expected, atol=1e-12
    )


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_posterior_matches_reference(categorical_data, k, alpha):
    X, y = categorical_data
    X_train, y_train, X_test = X[:300], y[:300], X[300:]
    model = KDB(k=k, alpha=alpha).fit(X_train, y_train)
    expected = _reference_proba(
        X_train, y_train, X_test, _as_lists(model.parents_), alpha
    )
    np.testing.assert_allclose(model.predict_proba(X_test), expected, atol=1e-10)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_k0_is_naive_bayes(categorical_data, alpha):
    # Proposition 1 of Sahami (1996, p. 336): the 0-dependence KDB is Naive Bayes
    X, y = categorical_data
    rng = np.random.RandomState(0)
    X = np.column_stack(
        [X, rng.normal(size=len(y)) + y, rng.randint(0, 2, size=len(y))]
    )
    model = KDB(k=0, alpha=alpha).fit(X, y)
    nb = MixedNB(alpha=alpha, var_prior_weight=1.0).fit(X, y)
    assert _as_lists(model.parents_) == [[]] * X.shape[1]
    assert not model._tables.n_parents.any()
    np.testing.assert_allclose(model.predict_proba(X), nb.predict_proba(X), atol=1e-12)
    expected = _reference_proba(X[:, :5], y, X[:, :5], [[]] * 5, alpha)
    model = KDB(k=0, alpha=alpha).fit(X[:, :5], y)
    np.testing.assert_allclose(model.predict_proba(X[:, :5]), expected, atol=1e-12)


def test_gaussian_child_estimates():
    # Two discrete parents (3 and 2 values), Gaussian child placed last
    rng = np.random.RandomState(0)
    n = 600
    y = rng.randint(0, 2, size=n)
    x0 = np.where(rng.rand(n) < 0.8, y, 2)
    x1 = np.where(rng.rand(n) < 0.7, y, 1 - y)
    # P(x0 == x1 | y) = 0.56 in both classes: x2 is informative only jointly
    x2 = 2.0 * (x0 == x1) + rng.normal(scale=0.3, size=n)
    X = np.column_stack([x0, x1, x2])
    model = KDB(k=2, var_prior_weight=2.0).fit(X, y)
    np.testing.assert_array_equal(model.feature_order_, [0, 1, 2])
    pa = model.parents_[2]
    assert sorted(pa.tolist()) == [0, 1]

    cond = model._conditional(2)
    eps = 1e-9 * x2.var()
    assert len(cond["keys"]) > 6
    for k, key in enumerate(cond["keys"]):
        c, *values = np.unravel_index(key, (2, *(X[:, pa].max(axis=0).astype(int) + 1)))
        rows = (y == c) & np.all(X[:, pa] == values, axis=1)
        n_k = rows.sum()
        var = (n_k * (x2[rows].var() + eps) + 2.0 * (x2.var() + eps)) / (n_k + 2.0)
        assert cond["count"][k] == n_k
        np.testing.assert_allclose(cond["mean"][k], x2[rows].mean())
        np.testing.assert_allclose(cond["var"][k], var)


# --- Unseen states ----------------------------------------------------------


def test_unseen_parent_category_is_uninformed(categorical_data):
    X, y = categorical_data
    model = KDB(k=X.shape[1] - 1).fit(X, y)
    first = model.feature_order_[0]
    X_test = X[:3].copy()
    X_test[:, first] = 7  # never seen; every other attribute has it as parent
    jll = model._joint_log_likelihood(X_test)
    first_nb = MixedNB().fit(X[:, [first]], y)
    cards = model.backoff_._cat_cardinalities
    children = [i for i in range(X.shape[1]) if i != first]
    expected = first_nb._joint_log_likelihood(X_test[:, [first]])
    expected -= np.log(cards[children]).sum()
    np.testing.assert_allclose(jll, expected, atol=1e-12)
    assert np.all(np.isfinite(model.predict_proba(X_test)))


def test_unseen_state_gaussian_child_is_pooled():
    rng = np.random.RandomState(0)
    y = np.array([0] * 60 + [1] * 60)
    x1 = np.where(y == 0, np.array([0, 1, 2] * 40), np.array([0, 1] * 60))
    x2 = rng.normal(size=120)
    X = np.column_stack([y, x1, x2])  # attribute 0 is a copy of the class
    model = KDB(k=2).fit(X, y)
    np.testing.assert_array_equal(model.feature_order_, [0, 1, 2])
    assert _as_lists(model.parents_)[:2] == [[], [0]]
    assert sorted(model.parents_[2].tolist()) == [0, 1]

    # Neither (y=0, x0=1) nor (y=1, x0=1, x1=2) was observed: the Gaussian
    # child takes the pooled Gaussian in both classes
    X_test = np.array([[1, 2, 0.3]])
    var = x2.var() + 1e-9 * x2.var()
    pooled = -0.5 * ((0.3 - x2.mean()) ** 2 / var + np.log(2 * np.pi * var))
    expected = np.log(0.5) + pooled
    expected += np.log([(0 + 1) / (60 + 2), (60 + 1) / (60 + 2)])  # P(x0 | y)
    expected += np.log([1 / 3, (0 + 1) / (60 + 3)])  # P(x1 | y, x0)
    np.testing.assert_allclose(
        model._joint_log_likelihood(X_test)[0], expected, atol=1e-10
    )


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
    model = KDB(
        k=2,
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


@pytest.mark.parametrize("k", [-1, 1.5, True, "1", None])
def test_invalid_k(k):
    X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="k"):
        KDB(k=k).fit(X, y)


def test_numpy_integer_k(categorical_data):
    X, y = categorical_data
    model = KDB(k=np.int64(2)).fit(X, y)
    assert _as_lists(model.parents_) == _as_lists(KDB(k=2).fit(X, y).parents_)


def test_prediction_memory_does_not_grow_with_the_parent_states():
    """Each conditional is evaluated only at the augmented states of the
    classes, not at every observed one."""
    import tracemalloc

    rng = np.random.RandomState(0)
    X = rng.randint(0, 6, size=(3000, 8))
    y = rng.randint(0, 2, size=3000)
    model = KDB(k=5, categorical_features=list(range(8))).fit(X, y)
    assert np.diff(model._tables.state_off).max() > 1000
    tracemalloc.start()
    model.predict_proba(X)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 50 * 2**20


def test_high_cardinality_matches_reference():
    """Children of hundreds of values under two parents keep only their
    non-empty (state, value) cells, with the same posterior."""
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, size=n)
    a0 = rng.randint(0, 120, size=n)
    a1 = np.where(rng.rand(n) < 0.7, (a0 + y) % 90, rng.randint(0, 90, size=n))
    a2 = np.where(rng.rand(n) < 0.7, (a0 + a1) % 100, rng.randint(0, 100, size=n))
    a3 = rng.randint(0, 3, size=n)
    X = np.column_stack([a0, a1, a2, a3])
    X = np.column_stack([np.unique(col, return_inverse=True)[1] for col in X.T])
    model = KDB(k=2, categorical_features=range(4)).fit(X, y)
    assert max(len(p) for p in model.parents_) == 2
    assert (np.diff(model._tables.cell_off) > 0).any()
    X_test = np.vstack([X, np.column_stack([rng.permutation(c) for c in X.T])])
    expected = _reference_proba(X, y, X_test, model.parents_)
    np.testing.assert_allclose(model.predict_proba(X_test), expected, atol=1e-10)


def test_too_many_parent_states_raise_a_clear_error():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 7, size=(300, 30))
    y = rng.randint(0, 2, size=300)
    with pytest.raises(ValueError, match="64-bit index"):
        KDB(k=29, categorical_features=list(range(30))).fit(X, y)


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_n_jobs_gives_the_same_model(categorical_data, n_jobs):
    X, y = categorical_data
    serial = KDB(k=2).fit(X, y)
    threaded = KDB(k=2, n_jobs=n_jobs).fit(X, y)
    assert _as_lists(threaded.parents_) == _as_lists(serial.parents_)
    np.testing.assert_allclose(threaded.predict_proba(X), serial.predict_proba(X))


def test_parentless_gaussians_are_those_of_naive_bayes():
    """The variance smoothing of MixedNB uses the largest variance of all the
    Gaussian attributes, which matters for an attribute on a small scale."""
    rng = np.random.RandomState(0)
    n = 200
    y = rng.randint(0, 2, size=n)
    X = np.column_stack(
        [1e4 * rng.normal(size=n) + 1e4 * y, 1e-4 * rng.normal(size=n) + 1e-4 * y]
    )
    model = KDB(k=0).fit(X, y)
    nb = MixedNB(var_prior_weight=1.0).fit(X, y)
    np.testing.assert_allclose(
        model._conditional(1)["var"], nb.estimators_["gaussian"].var_[:, 1]
    )
    np.testing.assert_allclose(
        model.predict_log_proba(X), nb.predict_log_proba(X), atol=1e-12
    )
