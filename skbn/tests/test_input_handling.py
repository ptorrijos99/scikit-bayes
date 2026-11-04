"""Tests for the typing and encoding of the attributes shared by all models."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.naive_bayes import BernoulliNB, CategoricalNB

from skbn import ALR, KDB, TAN, AnDE, AnJE, MixedNB, WeightedAnDE

ENSEMBLES = [
    MixedNB(),
    AnDE(n_dependence=1),
    AnDE(n_dependence=2),
    AnDE(n_dependence=1, mean="geometric"),
    TAN(),
    KDB(k=2),
    WeightedAnDE(n_dependence=1, weight_scope="factor", weight_level=4),
]


def _discrete_data(seed=0, n=300):
    """Three discrete attributes with codes 0..2 and a class that depends on
    the first two."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3, size=(n, 3))
    y = ((X[:, 0] + X[:, 1]) % 3 == 0).astype(int)
    flip = rng.rand(n) < 0.1
    return X, np.where(flip, 1 - y, y)


@pytest.mark.parametrize("values", [(1, 2), (0.3, 0.7), (-2, -1), (5, 9)])
def test_two_valued_feature_keeps_its_values_apart(values):
    """Any two-valued attribute is the categorical one with codes 0 and 1."""
    rng = np.random.RandomState(0)
    codes = rng.randint(0, 2, size=(200, 1))
    y = np.where(rng.rand(200) < 0.9, codes[:, 0], 1 - codes[:, 0])
    X = np.where(codes == 0, values[0], values[1]).astype(float)

    model = MixedNB().fit(X, y)
    assert model.feature_types_["bernoulli"] == [0]
    ref = CategoricalNB().fit(codes, y)
    assert_allclose(model.predict_proba(X), ref.predict_proba(codes))


def test_bernoulli_threshold_keeps_bernoulli_nb_convention():
    """Binary data on both sides of 0 is binarised at 0, as BernoulliNB does."""
    rng = np.random.RandomState(1)
    X = rng.randint(0, 2, size=(100, 4)).astype(float)
    y = rng.randint(0, 2, size=100)
    model = MixedNB(bernoulli_features=[0, 1, 2, 3]).fit(X, y)
    assert_allclose(model.bernoulli_thresholds_, 0.0)
    X_test = rng.randn(20, 4)
    assert_allclose(
        model.predict_proba(X_test), BernoulliNB().fit(X, y).predict_proba(X_test)
    )


@pytest.mark.parametrize("estimator", ENSEMBLES, ids=repr)
def test_categorical_codes_are_labels(estimator):
    """Relabelling the categories by an increasing map changes nothing: they
    are values, not positions, so negative or sparse codes need no care."""
    X, y = _discrete_data()
    relabel = np.array([-7, 40, 2_000_000])
    X_relabelled = relabel[X]
    declared = {"categorical_features": [0, 1, 2]}
    base = estimator.set_params(**declared)
    p_codes = base.fit(X, y).predict_proba(X)
    p_relabelled = base.fit(X_relabelled, y).predict_proba(X_relabelled)
    assert_allclose(p_codes, p_relabelled)


def test_sparse_codes_do_not_inflate_the_tables():
    X, y = _discrete_data()
    X = np.array([0, 1, 2_000_000])[X]
    model = MixedNB(categorical_features=[0, 1, 2]).fit(X, y)
    assert [len(c) for c in model.categories_] == [3, 3, 3]
    assert all(
        t.shape[1] == 3 for t in model.estimators_["categorical"].category_count_
    )


def test_unseen_category_is_marginalised():
    """A child value never seen in training contributes nothing, so the
    posterior is that of the model without the attribute."""
    X, y = _discrete_data()
    X_test = X[:20].astype(float)
    X_test[:, 2] = [3, 50, -1, 1.5] * 5
    full = MixedNB(categorical_features=[0, 1, 2]).fit(X, y)
    reduced = MixedNB(categorical_features=[0, 1]).fit(X[:, :2], y)
    assert_allclose(full.predict_proba(X_test), reduced.predict_proba(X_test[:, :2]))


@pytest.mark.parametrize("estimator", ENSEMBLES[1:], ids=repr)
def test_unseen_category_is_not_an_edge_category(estimator):
    """An unseen value is not read as the closest category of the training
    range (the clipping of earlier versions)."""
    X, y = _discrete_data()
    model = estimator.set_params(categorical_features=[0, 1, 2]).fit(X, y)
    X_unseen, X_edge = X[:30].copy(), X[:30].copy()
    X_unseen[:, 2], X_edge[:, 2] = 50, 2
    assert not np.allclose(model.predict_proba(X_unseen), model.predict_proba(X_edge))
    assert np.all(np.isfinite(model.predict_proba(X_unseen)))


def test_constant_gaussian_feature_is_ignored():
    """A declared Gaussian attribute without variance is ignored instead of
    making the posterior uniform."""
    X, y = _discrete_data()
    X_const = np.column_stack([np.ones(len(X)), X])
    model = MixedNB(gaussian_features=[0], categorical_features=[1, 2, 3])
    model.fit(X_const, y)
    assert model.feature_types_["gaussian"] == []
    ref = MixedNB(categorical_features=[0, 1, 2]).fit(X, y)
    assert_allclose(model.predict_proba(X_const), ref.predict_proba(X))


def test_boolean_masks_select_features():
    X, y = _discrete_data()
    by_index = MixedNB(categorical_features=[0, 2]).fit(X, y)
    by_mask = MixedNB(categorical_features=[True, False, True]).fit(X, y)
    assert by_mask.feature_types_ == by_index.feature_types_
    with pytest.raises(ValueError, match="boolean mask"):
        MixedNB(categorical_features=[True, False]).fit(X, y)
    with pytest.raises(ValueError, match="integer indices"):
        MixedNB(categorical_features=[0.0, 2.0]).fit(X, y)


@pytest.mark.parametrize(
    "estimator", [AnDE(n_bins=4), TAN(n_bins=4), KDB(n_bins=4)], ids=repr
)
def test_parents_and_children_share_the_attribute_types(estimator):
    """A negative integer-valued attribute is Gaussian as a child, so as a
    parent it is binned instead of label-encoded into hundreds of codes."""
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.randint(-500, 500, 400), rng.randint(0, 3, 400)])
    y = (X[:, 0] > 0).astype(int)
    model = estimator.fit(X.astype(float), y)
    assert model.backoff_.feature_types_["gaussian"] == [0]
    assert model._encoder.cards_[0] <= 4


def test_many_valued_integer_attribute_is_continuous():
    """An integer-valued attribute with many distinct values (an amount, an
    age) is Gaussian, not a categorical one with a cell per value."""
    rng = np.random.RandomState(0)
    amount = rng.randint(250, 1700, 200).astype(float)
    codes = rng.randint(0, 4, 200)
    model = MixedNB().fit(np.column_stack([amount, codes]), codes % 2)
    assert model.feature_types_ == {
        "gaussian": [0],
        "categorical": [1],
        "bernoulli": [],
    }


@pytest.mark.parametrize("estimator", [AnJE, ALR], ids=lambda e: e.__name__)
def test_join_models_accept_masks_and_arrays(estimator):
    X, y = _discrete_data()
    X = np.column_stack([X, np.random.RandomState(1).normal(size=len(X))])
    by_list = estimator(gaussian_features=[3], categorical_features=[0, 1, 2])
    by_mask = estimator(
        gaussian_features=np.array([False, False, False, True]),
        categorical_features=np.array([0, 1, 2]),
    )
    assert_allclose(
        by_list.fit(X, y).predict_proba(X), by_mask.fit(X, y).predict_proba(X)
    )
