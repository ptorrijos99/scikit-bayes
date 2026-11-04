"""Tests for skbn.mixed_nb module."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB

from skbn.mixed_nb import MixedNB

# The check_estimator() function is a rigorous test suite that ensures
# full compatibility with scikit-learn's API. Initially, it is marked
# as skipped. You should uncomment the decorator to run it once the basic
# implementation is stable. Passing this check is the main goal.
#


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


@pytest.mark.parametrize(
    "reference, make_X",
    [
        (GaussianNB, lambda rng: rng.normal(size=(200, 3))),
        (CategoricalNB, lambda rng: rng.integers(0, 4, size=(200, 3))),
        (BernoulliNB, lambda rng: rng.integers(0, 2, size=(200, 3))),
    ],
    ids=["gaussian", "categorical", "bernoulli"],
)
def test_equivalence_with_sklearn_naive_bayes(reference, make_X):
    """On a single attribute type, MixedNB reproduces the scikit-learn Naive Bayes."""
    rng = np.random.default_rng(0)
    X = make_X(rng)
    y = (X[:, 0] + X[:, 1] > np.median(X[:, 0] + X[:, 1])).astype(int)
    params = {} if reference is GaussianNB else {"alpha": 1.0}
    expected = reference(**params).fit(X, y).predict_proba(X)
    got = MixedNB(**params).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)


def _searchsorted_codes(col, categories):
    pos = np.minimum(np.searchsorted(categories, col), len(categories) - 1)
    known = categories[pos] == col
    return np.where(known, pos, 0), ~known


@pytest.mark.parametrize(
    "make_col",
    [
        lambda rng, n: rng.randint(-5, 8, n),  # small integer range: table
        lambda rng, n: rng.randint(0, 4, n).astype(float),  # integral floats
        lambda rng, n: rng.randint(0, 6, n).astype(np.uint8),
        lambda rng, n: rng.randint(0, 10**9, n),  # wide range: sorted search
        lambda rng, n: rng.randint(-3, 3, n) * 0.5,  # non-integral values
        lambda rng, n: rng.normal(size=n),
    ],
)
def test_category_encoding_matches_unique_and_searchsorted(make_col):
    from skbn._core import _encode_values, _unique_values

    rng = np.random.RandomState(0)
    for n in (1, 7, 300):
        col = make_col(rng, n)
        categories = _unique_values(col)
        reference = np.unique(col)
        assert categories.dtype == reference.dtype
        np.testing.assert_array_equal(categories, reference)
        unseen = np.array([-100, 250, 10**9 + 1])
        if col.dtype.kind == "f":
            unseen = np.array([0.25, -1e300, np.inf, np.nan])
        test = np.concatenate([col, unseen.astype(col.dtype), make_col(rng, 50)])
        codes, mask = _encode_values(test, categories)
        expected_codes, expected_mask = _searchsorted_codes(test, reference)
        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(mask, expected_mask)
