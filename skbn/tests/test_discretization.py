"""Tests for the supervised discretizers."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from sklearn.datasets import load_iris

from skbn.discretization import DecisionTreeDiscretizer, MDLPDiscretizer


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_decision_tree_discretizer_basic(iris_data):
    X, y = iris_data
    disc = DecisionTreeDiscretizer(n_bins=4, random_state=42)
    X_disc = disc.fit_transform(X, y)

    assert X_disc.shape == X.shape
    assert np.all(X_disc >= 0)
    assert np.all(X_disc <= 3)


def test_decision_tree_discretizer_fallback_single():
    # Create data where a feature has no correlation with y
    X = np.random.RandomState(42).randn(100, 2)
    y = np.ones(100)  # All same class, no split possible

    disc = DecisionTreeDiscretizer(n_bins=5, fallback="single", random_state=42)
    X_disc = disc.fit_transform(X, y)

    # Both features should fallback to single bin 0
    assert np.all(X_disc == 0)


def test_decision_tree_discretizer_fallback_quantile():
    # Create data where a feature has no correlation with y
    X = np.random.RandomState(42).randn(100, 2)
    y = np.ones(100)  # All same class, no split possible

    disc = DecisionTreeDiscretizer(n_bins=5, fallback="quantile", random_state=42)
    X_disc = disc.fit_transform(X, y)

    # Should be binned into up to 5 bins using KBinsDiscretizer
    assert len(np.unique(X_disc[:, 0])) > 1
    assert np.max(X_disc) <= 4


def test_decision_tree_discretizer_fallback_uniform():
    X = np.random.RandomState(42).randn(100, 2)
    y = np.ones(100)

    disc = DecisionTreeDiscretizer(n_bins=5, fallback="uniform", random_state=42)
    X_disc = disc.fit_transform(X, y)

    assert len(np.unique(X_disc[:, 0])) > 1
    assert np.max(X_disc) <= 4


def test_decision_tree_discretizer_nan_handling():
    X = np.array(
        [[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan], [5.0, 6.0], [np.nan, np.nan]]
    )
    y = np.array([0, 1, 0, 1, 0])

    disc = DecisionTreeDiscretizer(n_bins=3, random_state=42)
    X_disc = disc.fit_transform(X, y)

    assert X_disc.shape == X.shape
    assert not np.isnan(X_disc).any()


def test_decision_tree_discretizer_invalid_params():
    with pytest.raises(ValueError, match="n_bins"):
        DecisionTreeDiscretizer(n_bins=1).fit([[1], [2]], [0, 1])

    with pytest.raises(ValueError, match="fallback"):
        DecisionTreeDiscretizer(fallback="invalid").fit([[1], [2]], [0, 1])


# --- MDLPDiscretizer ---------------------------------------------------------


def test_mdlp_iris_matches_published_cut_points():
    from sklearn.datasets import load_iris

    from skbn import MDLPDiscretizer

    X, y = load_iris(return_X_y=True)
    cuts = MDLPDiscretizer().fit(X, y).cut_points_
    np.testing.assert_allclose(cuts[2], [2.45, 4.75])
    np.testing.assert_allclose(cuts[3], [0.8, 1.75])


def test_mdlp_collapses_uninformative_feature():
    from skbn import MDLPDiscretizer

    rng = np.random.RandomState(0)
    X = np.c_[rng.normal(size=2000), rng.normal(size=2000)]
    y = (X[:, 0] > 0.3).astype(int)
    disc = MDLPDiscretizer().fit(X, y)
    assert len(disc.cut_points_[0]) >= 1
    assert len(disc.cut_points_[1]) == 0
    assert np.all(disc.transform(X)[:, 1] == 0)


def test_mdlp_handles_nan_and_max_bins():
    from skbn import MDLPDiscretizer

    rng = np.random.RandomState(1)
    X = rng.uniform(size=(500, 1))
    y = np.digitize(X[:, 0], [0.2, 0.4, 0.6, 0.8])
    X[::17, 0] = np.nan
    disc = MDLPDiscretizer(max_bins=3).fit(X, y)
    assert len(disc.cut_points_[0]) <= 2
    out = disc.transform(X)
    assert np.isfinite(out).all()


def test_mdlp_invalid_max_bins():
    from skbn import MDLPDiscretizer

    with pytest.raises(ValueError, match="max_bins"):
        MDLPDiscretizer(max_bins=1).fit([[1.0], [2.0]], [0, 1])


# --- Feature names and pandas output ------------------------------------------


@pytest.mark.parametrize(
    "Discretizer", [DecisionTreeDiscretizer, MDLPDiscretizer], ids=lambda c: c.__name__
)
def test_discretizer_feature_names_and_pandas_output(Discretizer, iris_data):
    pd = pytest.importorskip("pandas")
    X, y = iris_data
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame(X, columns=names)

    disc = Discretizer().set_output(transform="pandas")
    out = disc.fit_transform(df, y)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == names
    assert out.index.equals(df.index)
    np.testing.assert_array_equal(out.to_numpy(), Discretizer().fit_transform(X, y))
    np.testing.assert_array_equal(disc.get_feature_names_out(), names)

    # Without feature names in the input the names are generated
    disc = Discretizer().set_output(transform="pandas").fit(X, y)
    expected = ["x0", "x1", "x2", "x3"]
    np.testing.assert_array_equal(disc.get_feature_names_out(), expected)
    assert list(disc.transform(X).columns) == expected


# --- Sample weights -----------------------------------------------------------


@pytest.mark.parametrize(
    "Discretizer, kwargs",
    [
        (MDLPDiscretizer, {}),
        (DecisionTreeDiscretizer, {"random_state": 0}),
        (DecisionTreeDiscretizer, {"fallback": "quantile", "random_state": 0}),
        (DecisionTreeDiscretizer, {"fallback": "uniform", "random_state": 0}),
    ],
    ids=[
        "MDLPDiscretizer",
        "DecisionTreeDiscretizer",
        "DecisionTreeDiscretizer-quantile",
        "DecisionTreeDiscretizer-uniform",
    ],
)
def test_discretizer_sample_weight_equivalence(Discretizer, kwargs, iris_data):
    X, y = iris_data
    rng = np.random.RandomState(7)
    sw = rng.choice([0, 1, 2, 3], size=len(y))
    X_rep = np.repeat(X, sw, axis=0)
    y_rep = np.repeat(y, sw, axis=0)

    weighted = Discretizer(**kwargs).fit(X, y, sample_weight=sw)
    repeated = Discretizer(**kwargs).fit(X_rep, y_rep)
    np.testing.assert_array_equal(weighted.transform(X), repeated.transform(X))


@pytest.mark.parametrize(
    "Discretizer", [DecisionTreeDiscretizer, MDLPDiscretizer], ids=lambda c: c.__name__
)
def test_discretizer_sample_weight_validation(Discretizer, iris_data):
    X, y = iris_data
    with pytest.raises(ValueError, match="non-negative"):
        Discretizer().fit(X, y, sample_weight=np.full(len(y), -1.0))
    # Recent scikit-learn rejects all-zero weights itself, with its own message
    with pytest.raises(ValueError, match="positive sum|at least one non-zero"):
        Discretizer().fit(X, y, sample_weight=np.zeros(len(y)))


def test_decision_tree_discretizer_fallback_quantile_weighted():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 2)
    y = np.ones(80, dtype=int)  # Single class triggers fallback
    sw = rng.choice([0, 1, 2, 3], size=80)
    X_rep = np.repeat(X, sw, axis=0)
    y_rep = np.repeat(y, sw, axis=0)

    for fallback in ("quantile", "uniform"):
        d_w = DecisionTreeDiscretizer(n_bins=4, fallback=fallback, random_state=0).fit(
            X, y, sample_weight=sw
        )
        d_r = DecisionTreeDiscretizer(n_bins=4, fallback=fallback, random_state=0).fit(
            X_rep, y_rep
        )
        np.testing.assert_array_equal(d_w.transform(X), d_r.transform(X))
