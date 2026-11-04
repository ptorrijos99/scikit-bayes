"""Tests of AnDE and WeightedAnDE on small mixed and XOR data sets.

The posteriors are checked exactly against a brute-force reference in
``test_ande_reference.py`` and the weights and gradients in
``test_ande_estimation.py``.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skbn import AnDE, MixedNB, WeightedAnDE

# Gaussian, Bernoulli and categorical attributes
X_MIXED = np.array(
    [[0.5, 0, 0], [-1.2, 1, 1], [0.6, 1, 2], [-0.1, 0, 0], [2.5, 1, 1], [-3.0, 0, 2]]
)
y_MIXED = np.array([0, 1, 1, 0, 1, 0])

# y = x0 XOR x1: no single attribute is informative
XOR = {
    "categorical": (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2),
        np.array([0, 1, 1, 0] * 2),
    ),
    "gaussian": (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]),
        np.array([0, 1, 1, 0]),
    ),
}


def test_ande_n0_is_mixed_nb_on_mixed_data():
    mnb = MixedNB().fit(X_MIXED, y_MIXED)
    ande = AnDE(n_dependence=0).fit(X_MIXED, y_MIXED)
    assert_allclose(ande.predict_proba(X_MIXED), mnb.predict_proba(X_MIXED))


@pytest.mark.parametrize("data", XOR)
@pytest.mark.parametrize("mean", ["arithmetic", "geometric"])
def test_a1de_solves_xor(data, mean):
    # Conditioning on one attribute makes the other determine the class; the
    # Gaussian attributes are binned at 0 when they act as parents
    X, y = XOR[data]
    model = AnDE(n_dependence=1, n_bins=2, mean=mean).fit(X, y)
    assert_array_equal(model.predict(X[:4]), y[:4])


@pytest.mark.parametrize("n_dependence", [0, 2])
def test_generative_weights_reproduce_ande_at_other_orders(n_dependence):
    for mean in ("arithmetic", "geometric"):
        ande = AnDE(n_dependence=n_dependence, mean=mean).fit(X_MIXED, y_MIXED)
        for scope in ("component", "factor"):
            model = WeightedAnDE(
                n_dependence=n_dependence, mean=mean, weight_scope=scope, max_iter=5
            ).fit(X_MIXED, y_MIXED)
            model.coef_theta_ = (
                np.zeros(model.n_weights_)
                if model._offset
                else np.ones(model.n_weights_)
            )
            assert_allclose(
                model.predict_proba(X_MIXED), ande.predict_proba(X_MIXED), atol=1e-10
            )


@pytest.mark.parametrize("mean", ["arithmetic", "geometric"])
@pytest.mark.parametrize("scope", ["component", "factor"])
def test_regularization_pulls_weights_towards_one(mean, scope):
    def deviation(l2_reg):
        model = WeightedAnDE(
            n_dependence=1,
            mean=mean,
            weight_scope=scope,
            weight_level=3,
            l2_reg=l2_reg,
            max_iter=50,
        ).fit(X_MIXED, y_MIXED)
        return np.abs(model.learned_weights_ - 1.0).mean()

    assert deviation(10.0) < deviation(1e-6)


def test_geometric_component_weights_are_non_negative():
    model = WeightedAnDE(
        n_dependence=1, mean="geometric", weight_level=3, l2_reg=0.0, max_iter=200
    ).fit(X_MIXED, y_MIXED)
    assert np.all(model.learned_weights_ >= 0)


@pytest.mark.parametrize("cls", [AnDE, WeightedAnDE])
def test_threads_give_the_same_model(cls):
    serial = cls(n_dependence=1, n_jobs=1).fit(X_MIXED, y_MIXED)
    parallel = cls(n_dependence=1, n_jobs=2).fit(X_MIXED, y_MIXED)
    assert_allclose(serial.predict_proba(X_MIXED), parallel.predict_proba(X_MIXED))
