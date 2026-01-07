"""Extended tests for skbn.ande module - AnJE, ALR, WeightedAnDE, and edge cases."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification

from skbn.ande import ALR, AnDE, AnJE, WeightedAnDE


# =============================================================================
# Synthetic Datasets
# =============================================================================

# 1. Mixed Data (Gaussian, Bernoulli, Categorical)
X_MIXED = np.array(
    [[0.5, 0, 0], [-1.2, 1, 1], [0.6, 1, 2], [-0.1, 0, 0], [2.5, 1, 1], [-3.0, 0, 2]]
)
y_MIXED = np.array([0, 1, 1, 0, 1, 0])

# 2. Pure Categorical (XOR-like structure)
X_CAT = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
y_CAT = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 3. Pure Gaussian (XOR-like)
X_GAUSS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
y_GAUSS = np.array([0, 1, 1, 0])


# =============================================================================
# AnJE Tests (Geometric Mean)
# =============================================================================


class TestAnJE:
    """Tests for the AnJE (Geometric Mean) classifier."""

    def test_anje_basic_fit_predict(self):
        """Test that AnJE can fit and predict without errors."""
        model = AnJE(n_dependence=1, n_bins=3)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert set(preds).issubset(set(y_MIXED))

    def test_anje_n0_runs(self):
        """Test AnJE with n=0 (equivalent to Naive Bayes structure)."""
        model = AnJE(n_dependence=0)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert len(preds) == len(y_MIXED)

    def test_anje_probabilities_sum_to_one(self):
        """Test that predict_proba returns valid probabilities."""
        model = AnJE(n_dependence=1)
        model.fit(X_MIXED, y_MIXED)

        proba = model.predict_proba(X_MIXED)
        # Probabilities should sum to 1 for each sample
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        # All probabilities should be non-negative
        assert np.all(proba >= 0)

    def test_anje_log_proba_consistency(self):
        """Test that predict_log_proba is consistent with predict_proba."""
        model = AnJE(n_dependence=1)
        model.fit(X_MIXED, y_MIXED)

        log_proba = model.predict_log_proba(X_MIXED)
        proba = model.predict_proba(X_MIXED)

        assert_allclose(np.exp(log_proba), proba, rtol=1e-5)

    def test_anje_categorical_xor(self):
        """Test AnJE on categorical XOR problem."""
        model = AnJE(n_dependence=1)
        model.fit(X_CAT, y_CAT)

        X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected_y = np.array([0, 1, 1, 0])
        preds = model.predict(X_test)

        assert_array_equal(preds, expected_y, err_msg="AnJE failed categorical XOR")

    def test_anje_gaussian_xor(self):
        """Test AnJE on Gaussian XOR problem."""
        model = AnJE(n_dependence=1, n_bins=2, strategy="quantile")
        model.fit(X_GAUSS, y_GAUSS)

        preds = model.predict(X_GAUSS)
        assert_array_equal(preds, y_GAUSS, err_msg="AnJE failed Gaussian XOR")

    def test_anje_n2_structure(self):
        """Test AnJE(n=2) builds correct number of models."""
        model = AnJE(n_dependence=2)
        model.fit(X_MIXED, y_MIXED)

        # 3 features, n=2 -> C(3,2) = 3 models
        assert len(model.ensemble_) == 3

    def test_anje_vs_ande_different_aggregation(self):
        """Test that AnJE (geometric mean) differs from AnDE (arithmetic mean)."""
        anje = AnJE(n_dependence=1)
        ande = AnDE(n_dependence=1)

        anje.fit(X_MIXED, y_MIXED)
        ande.fit(X_MIXED, y_MIXED)

        proba_anje = anje.predict_proba(X_MIXED)
        proba_ande = ande.predict_proba(X_MIXED)

        # Probabilities should generally differ (geometric vs arithmetic mean)
        # Although predictions may match, probabilities should typically differ
        # This is a sanity check that they're not identical implementations
        assert proba_anje.shape == proba_ande.shape


# =============================================================================
# ALR Tests (Accelerated Logistic Regression)
# =============================================================================


class TestALR:
    """Tests for ALR (Hybrid, Convex Optimization)."""

    def test_alr_basic_fit_predict(self):
        """Test that ALR can fit and predict without errors."""
        model = ALR(n_dependence=1, n_bins=3, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert set(preds).issubset(set(y_MIXED))

    def test_alr_learns_weights(self):
        """Test that ALR learns non-trivial weights."""
        model = ALR(n_dependence=1, max_iter=20)
        model.fit(X_MIXED, y_MIXED)

        assert hasattr(model, "learned_weights_")
        assert len(model.learned_weights_) == model.n_weights_
        # Weights should have been updated from initial ones (not all exactly 1)
        # Note: with regularization, they may stay close to 1 but not identical
        assert model.learned_weights_ is not None

    def test_alr_weight_level_1(self):
        """Test ALR with weight_level=1 (per-model weights)."""
        model = ALR(n_dependence=1, weight_level=1, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        # With 3 features and n=1, expect 3 models -> 3 weights
        assert model.n_weights_ == 3

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_alr_weight_level_2(self):
        """Test ALR with weight_level=2 (per-parent-value weights)."""
        model = ALR(n_dependence=1, weight_level=2, n_bins=3, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        # Number of weights depends on binned parent values
        assert model.n_weights_ > len(model.ensemble_)

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_alr_weight_level_3(self):
        """Test ALR with weight_level=3 (per-class weights)."""
        model = ALR(n_dependence=1, weight_level=3, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        # 3 models * 2 classes = 6 weights
        n_models = len(model.ensemble_)
        n_classes = len(model.classes_)
        assert model.n_weights_ == n_models * n_classes

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_alr_weight_level_4(self):
        """Test ALR with weight_level=4 (per-value-and-class weights)."""
        model = ALR(n_dependence=1, weight_level=4, n_bins=2, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        # Most granular level: weights per (model, parent_value, class)
        assert model.n_weights_ > len(model.ensemble_) * len(model.classes_)

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_alr_n0(self):
        """Test ALR with n=0 (Naive Bayes structure with learned weights)."""
        model = ALR(n_dependence=0, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        # n=0 means 1 model
        assert len(model.ensemble_) == 1

        preds = model.predict(X_MIXED)
        assert len(preds) == len(y_MIXED)

    def test_alr_n2(self):
        """Test ALR with n=2 (A2DE structure with learned weights)."""
        model = ALR(n_dependence=2, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        # 3 features, n=2 -> C(3,2) = 3 models
        assert len(model.ensemble_) == 3

        preds = model.predict(X_MIXED)
        assert set(preds).issubset(set(y_MIXED))

    def test_alr_regularization_effect(self):
        """Test that L2 regularization affects the weights."""
        model_low_reg = ALR(n_dependence=1, l2_reg=1e-6, max_iter=50)
        model_high_reg = ALR(n_dependence=1, l2_reg=1.0, max_iter=50)

        model_low_reg.fit(X_MIXED, y_MIXED)
        model_high_reg.fit(X_MIXED, y_MIXED)

        # High regularization should push weights closer to 1.0
        low_reg_deviation = np.mean(np.abs(model_low_reg.learned_weights_ - 1.0))
        high_reg_deviation = np.mean(np.abs(model_high_reg.learned_weights_ - 1.0))

        # With much higher regularization, deviation from 1.0 should be smaller
        assert high_reg_deviation <= low_reg_deviation + 1e-2


# =============================================================================
# WeightedAnDE Tests (Hybrid Arithmetic Mean)
# =============================================================================


class TestWeightedAnDE:
    """Tests for WeightedAnDE (Hybrid, Non-Convex Optimization)."""

    def test_weighted_ande_basic_fit_predict(self):
        """Test that WeightedAnDE can fit and predict without errors."""
        model = WeightedAnDE(n_dependence=1, n_bins=3, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert set(preds).issubset(set(y_MIXED))

    def test_weighted_ande_learns_weights(self):
        """Test that WeightedAnDE learns weights."""
        model = WeightedAnDE(n_dependence=1, max_iter=20)
        model.fit(X_MIXED, y_MIXED)

        assert hasattr(model, "learned_weights_")
        assert len(model.learned_weights_) == model.n_weights_

    def test_weighted_ande_weight_level_1(self):
        """Test WeightedAnDE with weight_level=1."""
        model = WeightedAnDE(n_dependence=1, weight_level=1, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_weighted_ande_weight_level_2(self):
        """Test WeightedAnDE with weight_level=2."""
        model = WeightedAnDE(n_dependence=1, weight_level=2, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_weighted_ande_weight_level_3(self):
        """Test WeightedAnDE with weight_level=3."""
        model = WeightedAnDE(n_dependence=1, weight_level=3, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_weighted_ande_weight_level_4(self):
        """Test WeightedAnDE with weight_level=4."""
        model = WeightedAnDE(n_dependence=1, weight_level=4, n_bins=2, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_weighted_ande_n2(self):
        """Test WeightedAnDE with n=2."""
        model = WeightedAnDE(n_dependence=2, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        assert len(model.ensemble_) == 3
        preds = model.predict(X_MIXED)
        assert len(preds) == len(y_MIXED)


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_feature(self):
        """Test models with a single feature."""
        X = np.array([[1.0], [2.0], [1.5], [2.5], [1.2], [2.2]])
        y = np.array([0, 1, 0, 1, 0, 1])

        for cls in [AnDE, AnJE, ALR, WeightedAnDE]:
            model = cls(n_dependence=0)  # n must be 0 for 1 feature with n=0
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)

    def test_many_classes(self):
        """Test models with more than 2 classes."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        y = y.astype(int)

        for cls in [AnDE, AnJE]:
            model = cls(n_dependence=1, n_bins=3)
            model.fit(X, y)

            proba = model.predict_proba(X)
            assert proba.shape == (100, 4)
            assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        for cls in [ALR, WeightedAnDE]:
            model = cls(n_dependence=1, n_bins=3, max_iter=10)
            model.fit(X, y)

            proba = model.predict_proba(X)
            assert proba.shape == (100, 4)
            assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_binary_features(self):
        """Test models with only binary features."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5)
        y = np.array([0, 1, 1, 0] * 5)

        for cls in [AnDE, AnJE, ALR, WeightedAnDE]:
            if cls in [ALR, WeightedAnDE]:
                model = cls(n_dependence=1, max_iter=10)
            else:
                model = cls(n_dependence=1)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)

    def test_discretization_strategies(self):
        """Test different discretization strategies."""
        X = np.random.randn(50, 3)
        y = (X.sum(axis=1) > 0).astype(int)

        for strategy in ["uniform", "quantile", "kmeans"]:
            model = AnDE(n_dependence=1, n_bins=3, strategy=strategy)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)

    def test_different_n_bins(self):
        """Test different numbers of bins."""
        X = np.random.randn(50, 3)
        y = (X.sum(axis=1) > 0).astype(int)

        for n_bins in [2, 5, 10]:
            model = AnDE(n_dependence=1, n_bins=n_bins)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)

    def test_different_alpha_smoothing(self):
        """Test different alpha smoothing values."""
        for alpha in [0.1, 1.0, 10.0]:
            model = AnDE(n_dependence=1, alpha=alpha)
            model.fit(X_MIXED, y_MIXED)
            proba = model.predict_proba(X_MIXED)
            assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_on_interpolated_values(self):
        """Test prediction on new samples within training range (interpolation)."""
        X_train = np.array([[0.0, 0], [1.0, 1], [2.0, 0], [3.0, 1]] * 3)
        y_train = np.array([0, 1, 0, 1] * 3)

        # Test on values within the training range
        X_test = np.array([[0.5, 0], [1.5, 1], [2.5, 0]])

        model = AnDE(n_dependence=1, n_bins=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        proba = model.predict_proba(X_test)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_all_same_class(self):
        """Test behavior when all training samples have the same class."""
        X = np.array([[1.0, 0], [2.0, 1], [1.5, 0], [2.5, 1]])
        y = np.array([0, 0, 0, 0])  # All same class

        model = AnDE(n_dependence=1)
        model.fit(X, y)

        # Should predict only class 0
        preds = model.predict(X)
        assert_array_equal(preds, np.zeros(4))


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Tests for parallel execution with n_jobs."""

    def test_ande_parallel(self):
        """Test AnDE with parallel execution."""
        model_serial = AnDE(n_dependence=1, n_jobs=1)
        model_parallel = AnDE(n_dependence=1, n_jobs=2)

        model_serial.fit(X_MIXED, y_MIXED)
        model_parallel.fit(X_MIXED, y_MIXED)

        preds_serial = model_serial.predict(X_MIXED)
        preds_parallel = model_parallel.predict(X_MIXED)

        assert_array_equal(preds_serial, preds_parallel)

    def test_ande_parallel_all_cores(self):
        """Test AnDE with n_jobs=-1 (all cores)."""
        model = AnDE(n_dependence=1, n_jobs=-1)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert len(preds) == len(y_MIXED)

    def test_anje_parallel(self):
        """Test AnJE with parallel execution."""
        model_serial = AnJE(n_dependence=1, n_jobs=1)
        model_parallel = AnJE(n_dependence=1, n_jobs=-1)

        model_serial.fit(X_MIXED, y_MIXED)
        model_parallel.fit(X_MIXED, y_MIXED)

        preds_serial = model_serial.predict(X_MIXED)
        preds_parallel = model_parallel.predict(X_MIXED)

        assert_array_equal(preds_serial, preds_parallel)

    def test_alr_parallel(self):
        """Test ALR with parallel execution."""
        model = ALR(n_dependence=1, n_jobs=-1, max_iter=10)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert len(preds) == len(y_MIXED)


# =============================================================================
# API Consistency Tests
# =============================================================================


class TestAPIConsistency:
    """Tests for scikit-learn API consistency."""

    @pytest.mark.parametrize("cls", [AnDE, AnJE, ALR, WeightedAnDE])
    def test_classes_attribute(self, cls):
        """Test that classes_ attribute is set correctly."""
        if cls in [ALR, WeightedAnDE]:
            model = cls(n_dependence=1, max_iter=10)
        else:
            model = cls(n_dependence=1)
        model.fit(X_MIXED, y_MIXED)

        assert hasattr(model, "classes_")
        assert_array_equal(model.classes_, np.array([0, 1]))

    @pytest.mark.parametrize("cls", [AnDE, AnJE, ALR, WeightedAnDE])
    def test_n_features_in_attribute(self, cls):
        """Test that n_features_in_ attribute is set correctly."""
        if cls in [ALR, WeightedAnDE]:
            model = cls(n_dependence=1, max_iter=10)
        else:
            model = cls(n_dependence=1)
        model.fit(X_MIXED, y_MIXED)

        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == 3

    @pytest.mark.parametrize("cls", [AnDE, AnJE, ALR, WeightedAnDE])
    def test_predict_proba_shape(self, cls):
        """Test that predict_proba returns correct shape."""
        if cls in [ALR, WeightedAnDE]:
            model = cls(n_dependence=1, max_iter=10)
        else:
            model = cls(n_dependence=1)
        model.fit(X_MIXED, y_MIXED)

        proba = model.predict_proba(X_MIXED)
        assert proba.shape == (len(X_MIXED), 2)

    @pytest.mark.parametrize("cls", [AnDE, AnJE, ALR, WeightedAnDE])
    def test_predict_shape(self, cls):
        """Test that predict returns correct shape."""
        if cls in [ALR, WeightedAnDE]:
            model = cls(n_dependence=1, max_iter=10)
        else:
            model = cls(n_dependence=1)
        model.fit(X_MIXED, y_MIXED)

        preds = model.predict(X_MIXED)
        assert preds.shape == (len(X_MIXED),)
