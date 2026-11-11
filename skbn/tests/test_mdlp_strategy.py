"""Tests for the supervised (MDL) discretization inside the models."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.pipeline import make_pipeline

from skbn import (
    ALR,
    KDB,
    TAN,
    AnDE,
    AnJE,
    GaussianAnJE,
    MDLPDiscretizer,
    WeightedAnDE,
)
from skbn import _core as core_module
from skbn._core import _AttributeEncoder


@pytest.fixture
def striped_data():
    """A continuous attribute whose class alternates over 30 intervals, a
    continuous attribute without information and a categorical one."""
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 30, 6000)
    y = (np.floor(x) % 2).astype(int)
    X = np.column_stack([x, rng.normal(size=6000), rng.randint(0, 4, 6000)])
    return X, y


def test_mdlp_strategy_is_the_discretizer_in_a_pipeline(striped_data):
    X, y = striped_data
    X = X[:, :2]
    pipeline = make_pipeline(
        MDLPDiscretizer(), AnJE(n_join=2, categorical_features=[0, 1])
    ).fit(X, y)
    model = AnJE(n_join=2, strategy="mdlp").fit(X, y)
    cuts = pipeline[0].cut_points_
    assert_array_equal(model._encoder.cards_, [len(c) + 1 for c in cuts])
    assert_allclose(model.predict_proba(X), pipeline.predict_proba(X), atol=1e-12)


def test_discretizer_codes_must_be_declared_categorical(striped_data):
    # More than 20 intervals are typed as continuous unless declared, and
    # binned again: the pipeline needs categorical_features, the strategy
    # does not
    X, y = striped_data
    X = X[:, :1]
    assert len(MDLPDiscretizer().fit(X, y).cut_points_[0]) > 20
    undeclared = make_pipeline(MDLPDiscretizer(), AnJE(n_join=1)).fit(X, y)
    declared = make_pipeline(
        MDLPDiscretizer(), AnJE(n_join=1, categorical_features=[0])
    ).fit(X, y)
    assert undeclared.score(X, y) < 0.7
    assert declared.score(X, y) == 1.0
    assert AnJE(n_join=1, strategy="mdlp").fit(X, y).score(X, y) == 1.0


@pytest.mark.parametrize(
    "model",
    [
        AnDE(n_dependence=1),
        WeightedAnDE(n_dependence=1, max_iter=5),
        GaussianAnJE(n_dependence=1),
        AnJE(n_join=2),
        ALR(n_join=2, max_iter=5, l2_reg=1.0),
        TAN(),
        KDB(k=2),
    ],
    ids=lambda m: type(m).__name__,
)
def test_mdlp_strategy_in_every_model(model, striped_data):
    X, y = striped_data
    model.set_params(strategy="mdlp", n_bins=2).fit(X, y)
    # 30 intervals whatever n_bins says, a single one for the attribute
    # without information, and the categorical attribute untouched
    assert_array_equal(model._encoder.cards_, [30, 1, 4])
    proba = model.predict_proba(X[:200])
    assert np.isfinite(proba).all()
    assert_allclose(proba.sum(axis=1), 1.0)


def test_code_width_follows_the_number_of_intervals(striped_data, monkeypatch):
    # The number of intervals is only known after the cuts are found
    X, y = striped_data
    encoder = _AttributeEncoder(n_bins=5, strategy="mdlp")
    assert encoder.fit_transform(X, y, dtype=None).dtype == np.uint16
    monkeypatch.setattr(
        core_module, "_mdlp_cuts", lambda x, y, n_classes: np.arange(1.0, 70000.0)
    )
    codes = _AttributeEncoder(n_bins=5, strategy="mdlp").fit_transform(
        X * 3000, y, dtype=None
    )
    assert codes.dtype == np.int32
    assert codes[:, 0].max() > 1 << 16


def test_max_bins_keeps_the_cuts_that_remove_most_entropy():
    # The cut at 2 comes first; of the two that follow, the one at 1
    # separates 2000 samples of two classes and the one at 3 only 60
    rng = np.random.RandomState(0)
    x = np.concatenate(
        [
            rng.uniform(0, 1, 1000),
            rng.uniform(1, 2, 1000),
            rng.uniform(2, 3, 2000),
            rng.uniform(3, 3.3, 60),
        ]
    )
    y = np.repeat([0, 1, 2, 0], [1000, 1000, 2000, 60])
    X = x.reshape(-1, 1)
    assert_allclose(MDLPDiscretizer().fit(X, y).cut_points_[0], [1, 2, 3], atol=0.01)
    capped = MDLPDiscretizer(max_bins=3).fit(X, y)
    assert_allclose(capped.cut_points_[0], [1, 2], atol=0.01)
