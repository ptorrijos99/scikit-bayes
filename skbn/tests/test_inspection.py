"""Tests for the public views of a fitted network: its arcs, its DOT text
and the weights of every attribute."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from skbn import (
    KDB,
    TAN,
    AnDE,
    GaussianAnJE,
    MDLPDiscretizer,
    WeightedKDB,
    WeightedTAN,
)


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


@pytest.fixture
def cars():
    """A table of labels: the class depends on two attributes, and the
    third follows the first."""
    rng = np.random.RandomState(0)
    n = 600
    price = rng.choice(["low", "medium", "high"], n)
    safety = rng.choice(["low", "high"], n)
    cost = np.where(rng.uniform(size=n) < 0.8, price, rng.choice(["low", "high"], n))
    y = np.where((safety == "high") & (price != "high"), "good", "bad")
    y = np.where(rng.uniform(size=n) < 0.1, "good", y)
    X = pd.DataFrame({"price": price, "safety": safety, "cost": cost})
    return X, y


def test_conditional_table_holds_the_smoothed_frequencies(cars):
    X, y = cars
    model = KDB(k=1, alpha=1.0).fit(X, y)
    child = "cost" if (0, 2) in model.edges_ else "price"
    parent = "price" if child == "cost" else "cost"
    table = model.conditional_table(child)
    assert table.parents == [X.columns.get_loc(parent)]
    assert table.feature_values.tolist() == ["high", "low", "medium"]
    assert table.mean is None and table.var is None
    assert_allclose(table.proba.sum(axis=1), 1.0)
    for row, (c, (pv,)) in enumerate(zip(table.classes, table.parent_values)):
        state = (y == c) & (X[parent] == pv)
        assert table.count[row] == state.sum()
        counts = [(state & (X[child] == v)).sum() for v in table.feature_values]
        assert_allclose(table.proba[row], (np.array(counts) + 1) / (state.sum() + 3))
    # By name, by index and as a frame, the same numbers
    frame = model.conditional_table(X.columns.get_loc(child), as_frame=True)
    assert_array_equal(frame.to_numpy(), table.proba)
    assert frame.index.names == ["class", parent]
    assert frame.columns.name == child
    assert (
        frame.loc[("good", "low"), "low"]
        == table.proba[
            (table.classes == "good") & (table.parent_values[:, 0] == "low"), 1
        ]
    )


def test_conditional_table_of_a_gaussian_child(iris):
    X, y = iris
    model = TAN(var_prior_weight=0.0).fit(X, y)
    child, parent = 3, int(model.parents_[3])
    table = model.conditional_table(child)
    assert table.proba is None and table.feature_values is None
    cuts = model.cut_points_[parent]
    assert len(cuts) == 4  # five bins
    bins = np.digitize(X.iloc[:, parent], cuts)
    for row, (c, (b,)) in enumerate(zip(table.classes, table.parent_values)):
        state = (y == c) & (bins == b)
        assert table.count[row] == state.sum()
        assert_allclose(table.mean[row], X.iloc[:, child][state].mean())
    frame = model.conditional_table(X.columns[child], as_frame=True)
    assert list(frame.columns) == ["mean", "var"]
    assert frame.index.names == ["class", X.columns[parent]]


def test_tables_reject_what_is_not_a_factor(iris):
    X, y = iris
    model = WeightedTAN(max_iter=5)
    with pytest.raises(NotFittedError):
        model.conditional_table(0)
    _fit(model, X.assign(constant=1.0), y)
    for method in (model.conditional_table, model.weight_table):
        with pytest.raises(ValueError, match="not the name of a feature"):
            method("petal")
        with pytest.raises(ValueError, match=r"index in \[0, 4\]"):
            method(5)
        with pytest.raises(ValueError, match="constant in training"):
            method("constant")
    with pytest.raises(ValueError, match="not the name of a feature"):
        TAN().fit(X.to_numpy(), y).conditional_table("petal width (cm)")


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_weight_table_names_the_rows_of_the_weights(cars, level):
    X, y = cars
    model = _fit(WeightedKDB(k=1, weight_level=level, max_iter=20), X, y)
    for i, name in enumerate(X.columns):
        table = model.weight_table(name)
        assert_array_equal(table.weights, model.feature_weights_[i])
        assert (table.classes is None) == (level < 3)
        if level >= 3:
            assert_array_equal(table.classes, ["bad", "good"])
        parents = [p for p, child in model.edges_ if child == i]
        if level == 1:
            assert table.feature_values is None and len(table.weights) == 1
        elif level < 4 or not parents:
            assert table.parents == []
            assert table.feature_values.tolist() == sorted(X[name].unique())
        else:  # One row per value of the attribute and of its parent, as seen
            assert table.parents == parents
            seen = set(zip(X[name], X.iloc[:, parents[0]]))
            rows = set(zip(table.feature_values, table.parent_values[:, 0]))
            assert rows == seen and len(table.weights) == len(seen)
        frame = model.weight_table(i, as_frame=True)
        assert_array_equal(frame.to_numpy(), table.weights)
        assert list(frame.columns) == (["weight"] if level < 3 else ["bad", "good"])


def test_weight_table_of_a_gaussian_child_has_its_parent_states(iris):
    X, y = iris
    model = _fit(WeightedTAN(weight_level=4, max_iter=20), X, y)
    child, parent = 3, int(model.parents_[3])
    table = model.weight_table(child)
    assert table.feature_values is None and table.parents == [parent]
    bins = np.digitize(X.iloc[:, parent], model.cut_points_[parent])
    assert sorted(table.parent_values[:, 0]) == sorted(np.unique(bins))
    assert model.weight_table(child, as_frame=True).index.names == [X.columns[parent]]


def test_cut_points_are_those_the_model_bins_with(iris):
    X, y = iris
    X = X.assign(grade=np.digitize(X.iloc[:, 2], [2.5, 4.9]))
    expected = MDLPDiscretizer().fit(X.iloc[:, :4], y).cut_points_
    for model in (
        TAN(strategy="mdlp"),
        AnDE(strategy="mdlp"),
        KDB(strategy="mdlp", continuous_children="discretized"),
    ):
        cuts = model.fit(X, y).cut_points_
        assert cuts[4] is None  # a categorical attribute has no bins
        for ours, theirs in zip(cuts, expected):
            assert_array_equal(ours, theirs)
    quantiles = TAN(n_bins=4).fit(X, y).cut_points_
    assert [len(c) for c in quantiles[:4]] == [3, 3, 3, 3]
    assert_allclose(quantiles[0], np.quantile(X.iloc[:, 0], [0.25, 0.5, 0.75]))
