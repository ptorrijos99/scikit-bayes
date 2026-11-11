"""Tests for the models that discretize their continuous attributes as
children too (``continuous_children="discretized"``)."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

from skbn import (
    KDB,
    TAN,
    AnDE,
    MDLPDiscretizer,
    WeightedAnDE,
    WeightedKDB,
    WeightedTAN,
)

MODELS = [
    AnDE(n_dependence=1),
    AnDE(n_dependence=2, mean="geometric"),
    WeightedAnDE(max_iter=15),
    TAN(),
    KDB(k=2),
    WeightedTAN(max_iter=15),
    WeightedKDB(k=2, max_iter=15, weight_level=2),
]


@pytest.fixture
def wine():
    X, y = load_wine(return_X_y=True)
    order = np.random.RandomState(0).permutation(len(y))
    return X[order[:130]], y[order[:130]], X[order[130:]]


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)


@pytest.mark.parametrize("model", MODELS, ids=str)
def test_mdlp_children_are_the_discretizer_in_a_pipeline(model, wine):
    # The model on MDL intervals of the original works: every attribute is
    # discretized first, and the model sees categories
    X, y, X_test = wine
    d = X.shape[1]
    ours = _fit(
        clone(model).set_params(strategy="mdlp", continuous_children="discretized"),
        X,
        y,
    )
    pipeline = _fit(
        make_pipeline(
            MDLPDiscretizer(),
            clone(model).set_params(categorical_features=list(range(d))),
        ),
        X,
        y,
    )
    assert_allclose(
        ours.predict_proba(X_test), pipeline.predict_proba(X_test), atol=1e-12
    )
    assert ours.feature_types_["gaussian"] == []


@pytest.mark.parametrize("model", [AnDE(), TAN(), KDB(k=2)], ids=str)
@pytest.mark.parametrize("strategy", ["quantile", "uniform"])
def test_binned_children_are_kbins_in_a_pipeline(model, strategy, wine):
    X, y, X_test = wine
    d = X.shape[1]
    ours = clone(model).set_params(
        strategy=strategy, n_bins=4, continuous_children="discretized"
    )
    kwargs = {"quantile_method": "linear"} if strategy == "quantile" else {}
    try:
        binner = KBinsDiscretizer(4, encode="ordinal", strategy=strategy, **kwargs)
        binner.fit(X)
    except TypeError:  # scikit-learn before 1.7 has no quantile_method
        binner = KBinsDiscretizer(4, encode="ordinal", strategy=strategy)
    pipeline = make_pipeline(
        binner, clone(model).set_params(categorical_features=list(range(d)))
    )
    assert_allclose(
        ours.fit(X, y).predict_proba(X_test),
        pipeline.fit(X, y).predict_proba(X_test),
        atol=1e-12,
    )


def test_only_the_continuous_attributes_are_binned(wine):
    # Labels, declared categories and binary attributes stay as they are;
    # a continuous attribute declared Gaussian is binned like the others
    X, y, X_test = wine

    def table(X):
        return pd.DataFrame(
            {
                "a": X[:, 0],
                "b": X[:, 6],
                "grade": np.where(X[:, 9] > 5, "strong", "light"),
                "level": np.digitize(X[:, 12], [500, 800, 1100]),
                "flag": X[:, 1] > 2,
            }
        )

    model = AnDE(
        strategy="mdlp", continuous_children="discretized", gaussian_features=[1]
    ).fit(table(X), y)
    assert model._binning.columns == [0, 1]
    types = model.feature_types_
    assert types["gaussian"] == [] and types["bernoulli"] == [4]
    assert sorted(types["categorical"]) == [0, 1, 2, 3]
    # The same model from the bins of the two columns, given as categories
    binned = table(X)
    binned_test = table(X_test)
    for column in ("a", "b"):
        cuts = MDLPDiscretizer().fit(table(X)[[column]], y).cut_points_[0]
        binned[column] = np.digitize(binned[column], cuts)
        binned_test[column] = np.digitize(binned_test[column], cuts)
    reference = AnDE(categorical_features=[0, 1]).fit(binned, y)
    assert_allclose(
        model.predict_proba(table(X_test)),
        reference.predict_proba(binned_test),
        atol=1e-12,
    )


@pytest.mark.parametrize("model", [AnDE(), TAN(), WeightedKDB(max_iter=10)], ids=str)
def test_values_beyond_the_training_range_take_the_end_bins(model, wine):
    X, y, X_test = wine
    model = _fit(
        clone(model).set_params(strategy="mdlp", continuous_children="discretized"),
        X,
        y,
    )
    low, high = X.min(axis=0), X.max(axis=0)
    assert_array_equal(
        model.predict_proba(np.vstack([low - 1e6, high + 1e6])),
        model.predict_proba(np.vstack([low, high])),
    )


def test_default_keeps_gaussian_children(wine):
    X, y, X_test = wine
    default = TAN().fit(X, y)
    assert default._binning is None
    assert len(default.feature_types_["gaussian"]) == X.shape[1]
    assert_array_equal(
        default.predict_proba(X_test),
        TAN(continuous_children="gaussian").fit(X, y).predict_proba(X_test),
    )
    # and a model fitted again without the option forgets its bins
    model = TAN(continuous_children="discretized").fit(X, y)
    assert model._binning is not None
    model.set_params(continuous_children="gaussian").fit(X, y)
    assert_array_equal(model.predict_proba(X_test), default.predict_proba(X_test))


def test_data_are_not_modified_and_the_model_pickles(wine):
    X, y, X_test = wine
    X_before, test_before = X.copy(), X_test.copy()
    model = AnDE(strategy="mdlp", continuous_children="discretized").fit(X, y)
    expected = model.predict_proba(X_test)
    assert_array_equal(X, X_before)
    assert_array_equal(X_test, test_before)
    restored = pickle.loads(pickle.dumps(model))
    assert_array_equal(restored.predict_proba(X_test), expected)


def test_discretized_children_without_continuous_attributes():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 3, size=(200, 4))
    y = (X[:, 0] + X[:, 1]) % 2
    model = KDB(k=2, continuous_children="discretized").fit(X, y)
    assert model._binning is None
    assert_array_equal(model.predict_proba(X), KDB(k=2).fit(X, y).predict_proba(X))
