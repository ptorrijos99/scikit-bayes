"""Tests for the public views of a fitted network: its arcs, its DOT text
and the weights of every attribute."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from skbn import KDB, TAN, AnDE, GaussianAnJE, WeightedKDB, WeightedTAN


@pytest.fixture
def iris():
    return load_iris(return_X_y=True, as_frame=True)


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


def test_edges_are_the_parents_of_tan_and_kdb(iris):
    X, y = iris
    tan = TAN(root=2).fit(X, y)
    assert tan.edges_ == [(p, i) for i, p in enumerate(tan.parents_) if p >= 0]
    assert len(tan.edges_) == X.shape[1] - 1  # a tree
    kdb = KDB(k=2).fit(X, y)
    assert kdb.edges_ == [(p, i) for i, pa in enumerate(kdb.parents_) for p in pa]
    assert KDB(k=0).fit(X, y).edges_ == []
    with pytest.raises(NotFittedError):
        TAN().edges_


def test_dot_holds_the_class_and_every_arc(iris):
    X, y = iris
    model = TAN().fit(X, y)
    names = list(X.columns)
    lines = model.to_dot().splitlines()
    assert lines[0] == "digraph {" and lines[-1] == "}"
    expected = [f'    "class" -> "{name}";' for name in names]
    expected += [f'    "{names[p]}" -> "{names[i]}";' for p, i in model.edges_]
    assert lines[1:-1] == expected

    # Without names in training, x0, x1, ...; given names are quoted
    model = KDB(k=1).fit(X.to_numpy(), y)
    assert '"class" -> "x3";' in model.to_dot()
    dot = model.to_dot(feature_names=['a "b"', "c", "d", "e"], class_name="y")
    assert '    "y" -> "a \\"b\\"";' in dot.splitlines()
    with pytest.raises(ValueError, match="4 names"):
        model.to_dot(feature_names=["a", "b"])


def test_dot_leaves_out_the_ignored_attributes():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    x = rng.normal(size=200) + y
    model = TAN().fit(np.column_stack([np.zeros(200), x, x + rng.normal(size=200)]), y)
    assert model.to_dot().splitlines()[1:-1] == [
        '    "class" -> "x1";',
        '    "class" -> "x2";',
        '    "x1" -> "x2";',
    ]


@pytest.mark.parametrize("Model", [WeightedTAN, WeightedKDB])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_weights_by_attribute_partition_the_learned_weights(iris, Model, level):
    X, y = iris
    X = X.to_numpy().copy()
    X[:, 3] = np.digitize(X[:, 3], [0.8, 1.7])  # a discrete attribute
    model = _fit(Model(weight_level=level, max_iter=10), X, y)
    n_classes = 3 if level >= 3 else 1
    assert model.prior_weights_.shape == (n_classes,)
    assert all(w.shape[1] == n_classes for w in model.feature_weights_)
    # The prior, the attributes in order and the untrained weight are all
    # the weights, in the order of learned_weights_
    parts = [model.prior_weights_] + [w.ravel() for w in model.feature_weights_]
    assert_array_equal(np.concatenate(parts + [[1.0]]), model.learned_weights_)
    if level == 1:
        assert [w.shape for w in model.feature_weights_] == [(1, 1)] * 4
    else:  # one row per value of the discrete attribute at least
        assert len(model.feature_weights_[3]) >= 3


def test_weights_of_an_ignored_attribute_are_empty():
    # A constant column has no factor: indexing learned_weights_ by column
    # would be off by one from there on
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    x = rng.normal(size=300) + y
    model = WeightedTAN(weight_level=1).fit(np.column_stack([np.zeros(300), x, x]), y)
    weights = model.feature_weights_
    assert weights[0].shape == (0, 1)
    assert_allclose([weights[1][0, 0], weights[2][0, 0]], model.learned_weights_[1:3])
    # The exact copy is discounted
    assert weights[2][0, 0] < 0.8 < weights[1][0, 0]


def _relabelled_data(seed):
    """x1 is x0 with its values renamed, so x2 has exactly the same
    conditional mutual information with both; the compiled sums give values
    one ulp apart on about half of the seeds."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, 60)
    x0 = rng.randint(0, 3, 60)
    x2 = (x0 + y + rng.randint(0, 2, 60)) % 3
    return np.column_stack([x0, (x0 + 1) % 3, x2]), y


def test_ties_between_edges_do_not_follow_the_rounding():
    # Each attribute keeps the parent that reached its best weight first: x2
    # hangs from x0, whatever the last bit of the two weights
    differ = 0
    for seed in range(40):
        X, y = _relabelled_data(seed)
        tan = TAN().fit(X, y)
        cmi = tan.conditional_mutual_info_
        differ += cmi[2, 0] != cmi[2, 1]
        assert_allclose(cmi[2, 0], cmi[2, 1], rtol=1e-12)
        assert_array_equal(tan.parents_, [-1, 0, 0])
        # KDB: x0 and x1 tie with the class too and are placed in that order;
        # x2 takes the one placed first
        kdb = KDB(k=1).fit(X, y)
        position = np.argsort(kdb.feature_order_)
        assert position[0] < position[1]
        if position[2] > position[1]:
            assert_array_equal(kdb.parents_[2], [0])
    assert differ > 0  # the case the tolerance is for did occur


@pytest.mark.parametrize(
    "model",
    [TAN(), KDB(k=2), WeightedTAN(max_iter=5), AnDE(), GaussianAnJE()],
    ids=lambda m: type(m).__name__,
)
def test_feature_types_are_those_of_the_backoff_model(model, iris):
    X, y = iris
    X = X.assign(wide=X.iloc[:, 3] > 1.6, grade=np.digitize(X.iloc[:, 2], [2.5, 4.9]))
    X["constant"] = 1.0
    with pytest.raises(NotFittedError):
        model.feature_types_
    types = _fit(model, X, y).feature_types_
    assert types == {"gaussian": [0, 1, 2, 3], "categorical": [5], "bernoulli": [4]}
    assert types is model.backoff_.feature_types_
