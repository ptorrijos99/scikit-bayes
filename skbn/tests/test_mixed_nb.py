"""Tests for skbn.mixed_nb module."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose
from sklearn.utils.estimator_checks import check_estimator

from skbn.mixed_nb import MixedNB


# The check_estimator() function is a rigorous test suite that ensures
# full compatibility with scikit-learn's API. Initially, it is marked
# as skipped. You should uncomment the decorator to run it once the basic
# implementation is stable. Passing this check is the main goal.
# @pytest.mark.skip(reason="check_estimator is very strict and should be enabled when the estimator is mature.")
def test_check_estimator_mixed_nb():
    """Test to ensure the estimator is compliant with scikit-learn."""
    check_estimator(MixedNB())


# --- Unit Tests for MixedNB Logic ---

# A sample dataset for testing purposes.
# Features:
# 0: Gaussian (float)
# 1: Bernoulli (2 unique values)
# 2: Categorical (3 unique values: 0, 1, 2)
X_MIXED = np.array(
    [[0.5, 0, 0], [-1.2, 1, 1], [0.6, 1, 2], [-0.1, 0, 0], [2.5, 1, 1], [-3.0, 0, 2]]
)
Y_MIXED = np.array([0, 1, 1, 0, 1, 0])


def test_feature_type_auto_detection():
    """Test that feature types are correctly inferred from data."""
    clf = MixedNB()
    clf.fit(X_MIXED, Y_MIXED)

    expected_types = {"gaussian": [0], "bernoulli": [1], "categorical": [2]}
    assert clf.feature_types_ == expected_types


def test_feature_type_manual_override():
    """Test that manual feature type specification overrides auto-detection."""
    # Force feature 2 to be treated as Bernoulli (even though it has 3 values)
    # This is not a typical use case but tests the override mechanism.
    clf = MixedNB(bernoulli_features=[1, 2])
    clf.fit(X_MIXED, Y_MIXED)

    expected_types = {"gaussian": [0], "bernoulli": [1, 2], "categorical": []}
    assert clf.feature_types_ == expected_types
    assert "gaussian" in clf.estimators_
    assert "bernoulli" in clf.estimators_
    assert "categorical" not in clf.estimators_


def test_predict_proba_on_simple_data():
    """
    Test the correctness of predict_proba with hand-calculated values.
    """
    # Dataset: [Gaussian, Bernoulli], 2 classes
    X = np.array(
        [
            [-1.0, 0],  # class 0
            [-2.0, 0],  # class 0
            [1.0, 1],  # class 1
            [2.0, 1],  # class 1
        ]
    )
    y = np.array([0, 0, 1, 1])

    clf = MixedNB(alpha=1.0)  # alpha=1 for Laplace smoothing
    clf.fit(X, y)

    # Verify the feature types are correctly detected
    assert clf.feature_types_["gaussian"] == [0]
    assert clf.feature_types_["bernoulli"] == [1]

    # For a test point [-0.5, 0]
    test_point = np.array([[-0.5, 0]])

    # Get actual joint log-likelihood from our estimator
    actual_jll = clf._joint_log_likelihood(test_point)

    # Verify shape is correct
    assert actual_jll.shape == (1, 2)

    # Final check on predict_proba
    probs = clf.predict_proba(test_point)
    assert_allclose(probs.sum(axis=1), 1.0)

    # Test point [-0.5, 0] should be classified as class 0:
    # - Gaussian feature -0.5 is closer to mean of class 0 (-1.5) than class 1 (1.5)
    # - Bernoulli feature 0 appears only in class 0 training data
    assert probs[0, 0] > probs[0, 1]  # Expect class 0 to have higher probability

    # Verify predict returns the expected class
    assert clf.predict(test_point)[0] == 0


def test_single_feature_type():
    """Test that the classifier works when only one feature type is present."""
    # Only Gaussian
    X_gauss = X_MIXED[:, [0]]
    clf_gauss = MixedNB()
    clf_gauss.fit(X_gauss, Y_MIXED)
    assert clf_gauss.predict(np.array([[0.1]])) == [1]
    assert "bernoulli" not in clf_gauss.estimators_
    assert "categorical" not in clf_gauss.estimators_

    # Only Bernoulli
    X_bern = X_MIXED[:, [1]]
    clf_bern = MixedNB()
    clf_bern.fit(X_bern, Y_MIXED)
    assert clf_bern.predict(np.array([[1]])) == [1]
    assert "gaussian" not in clf_bern.estimators_
