"""Tests for skbn.ande module."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose

from skbn.ande import AnDE
from skbn.mixed_nb import MixedNB

# --- Synthetic Datasets ---

# 1. Mixed Data (Gaussian, Bernoulli, Categorical)
# Feature 0: Gaussian (float)
# Feature 1: Bernoulli (0, 1)
# Feature 2: Categorical (0, 1, 2)
X_MIXED = np.array(
    [[0.5, 0, 0], [-1.2, 1, 1], [0.6, 1, 2], [-0.1, 0, 0], [2.5, 1, 1], [-3.0, 0, 2]]
)
y_MIXED = np.array([0, 1, 1, 0, 1, 0])

# 2. Pure Categorical (XOR-like structure to test dependencies)
# X1, X2 in {0, 1}. y = X1 XOR X2
X_CAT = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
y_CAT = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 3. Pure Gaussian (XOR-like)
# Covered by examples/plot_ande_xor.py, but good to have a smoke test here
X_GAUSS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
y_GAUSS = np.array([0, 1, 1, 0])


def test_ande_n0_equivalence_mixed():
    """
    Test that AnDE(n=0) is mathematically identical to MixedNB
    on mixed data types.
    """
    # Fit MixedNB
    mnb = MixedNB()
    mnb.fit(X_MIXED, y_MIXED)
    probs_mnb = mnb.predict_proba(X_MIXED)

    # Fit AnDE (n=0)
    ande = AnDE(n_dependence=0)
    ande.fit(X_MIXED, y_MIXED)
    probs_ande = ande.predict_proba(X_MIXED)

    # Check equality
    assert_allclose(probs_mnb, probs_ande, err_msg="AnDE(n=0) must match MixedNB")


def test_ande_n1_mixed_data_flow():
    """
    Test that AnDE(n=1) runs without errors on mixed data.
    Crucial check: The discretization logic must handle the mixed types
    (ignoring categorical/bernoulli columns, discretizing gaussian).
    """
    ande = AnDE(n_dependence=1, n_bins=3)
    ande.fit(X_MIXED, y_MIXED)

    # Check that predictions return valid class labels
    preds = ande.predict(X_MIXED)
    assert set(preds).issubset(set(y_MIXED))

    # Check structure
    # With 3 features and n=1, we expect 3 models in the ensemble
    assert len(ande.ensemble_) == 3


def test_ande_pure_categorical():
    """
    Test AnDE on pure categorical data.
    Since data is integers, the internal discretizer should leave it alone
    or map it consistently.
    """
    # A1DE (n=1) should be able to solve 2D XOR perfectly
    # (Condition on X1 -> X2 determines Y)
    ande = AnDE(n_dependence=1)
    ande.fit(X_CAT, y_CAT)

    # Predict on the unique combinations
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_y = np.array([0, 1, 1, 0])

    preds = ande.predict(X_test)
    assert_allclose(preds, expected_y, err_msg="A1DE failed to solve Categorical XOR")


def test_ande_pure_gaussian():
    """
    Test AnDE on pure gaussian data.
    The internal discretizer must actively bin the float features.
    """
    # Use n_bins=2. If we split at 0, we separate negative/positive.
    # Gaussian XOR is solvable if we discretize properly.
    ande = AnDE(n_dependence=1, n_bins=2, strategy="quantile")
    ande.fit(X_GAUSS, y_GAUSS)

    preds = ande.predict(X_GAUSS)
    assert_allclose(
        preds,
        y_GAUSS,
        err_msg="A1DE failed to solve Gaussian XOR with proper discretization",
    )


def test_ande_n2_logic():
    """
    Test that AnDE(n=2) builds the correct number of models.
    """
    # X_MIXED has 3 features.
    # Combinations of 2 parents from 3 features = 3C2 = 3 models.
    # Parents: (0,1), (0,2), (1,2)
    ande = AnDE(n_dependence=2)
    ande.fit(X_MIXED, y_MIXED)

    assert len(ande.ensemble_) == 3

    # Verify parents indices
    expected_parents = [(0, 1), (0, 2), (1, 2)]
    actual_parents = [tuple(m["parent_indices"]) for m in ande.ensemble_]
    assert set(expected_parents) == set(actual_parents)
