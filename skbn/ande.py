"""
Family of Averaged n-Dependence Estimators (AnDE) and Accelerated Logistic Regression (ALR).

This module implements a unified framework for n-dependence Bayesian classifiers
that supports mixed data types (continuous, categorical, binary) natively using
the "Super-Class" strategy.

It includes:
1.  **AnDE:** The classic AODE/A2DE generative model (Arithmetic Mean).
2.  **AnJE:** The generative model based on Geometric Mean.
3.  **ALR:** A hybrid discriminative model that optimizes weights for AnJE (Convex).
4.  **WeightedAnDE:** A hybrid discriminative model that optimizes weights for AnDE.

References
----------
.. [1] Webb, G. I., Boughton, J., & Wang, Z. (2005). Not so naive Bayes:
       Aggregating one-dependence estimators. Machine Learning, 58(1), 5-24.
.. [2] Zaidi, N. A., Webb, G. I., Carman, M. J., & Petitjean, F. (2017).
       Efficient parameter learning of Bayesian network classifiers.
       Machine Learning, 106(9-10), 1289-1329.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelBinarizer, LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from .mixed_nb import MixedNB

# --- Helper Functions for Parallelization ---
# Defined at module level to ensure picklability with joblib


def _fit_spode(
    parent_indices,
    X,
    y,
    parent_data,
    n_features,
    alpha,
    categorical_features=None,
    bernoulli_features=None,
    gaussian_features=None,
):
    """Fits a single SPODE (Sub-model) in parallel."""
    # A. Construct Y*
    if len(parent_indices) > 0:
        parents_vals = parent_data[:, parent_indices]
        # Efficient string signature
        y_augmented = [
            f"{label}|" + "|".join(map(str, row)) for label, row in zip(y, parents_vals)
        ]
    else:
        y_augmented = y

    le = LabelEncoder()
    y_augmented_enc = le.fit_transform(y_augmented)

    # B. Identify features for child NB
    child_indices = [i for i in range(n_features) if i not in parent_indices]

    if not child_indices:
        X_train_sub = np.zeros((X.shape[0], 1))
    else:
        X_train_sub = X[:, child_indices]

    sub_cat_indices = []
    sub_bern_indices = []
    sub_gauss_indices = []

    # Map global to local
    # child_indices contains the global indices effectively present in X_train_sub
    global_to_sub = {g_idx: s_idx for s_idx, g_idx in enumerate(child_indices)}

    if categorical_features is not None:
        for g_idx in categorical_features:
            if g_idx in global_to_sub:
                sub_cat_indices.append(global_to_sub[g_idx])

    if bernoulli_features is not None:
        for g_idx in bernoulli_features:
            if g_idx in global_to_sub:
                sub_bern_indices.append(global_to_sub[g_idx])

    if gaussian_features is not None:
        for g_idx in gaussian_features:
            if g_idx in global_to_sub:
                sub_gauss_indices.append(global_to_sub[g_idx])

    # D. Fit MixedNB
    sub_model = MixedNB(
        alpha=alpha,
        categorical_features=sub_cat_indices if sub_cat_indices else None,
        bernoulli_features=sub_bern_indices if sub_bern_indices else None,
        gaussian_features=sub_gauss_indices if sub_gauss_indices else None,
    )
    sub_model.fit(X_train_sub, y_augmented_enc)

    # D. Metadata
    decoded_classes = le.inverse_transform(sub_model.classes_)
    map_k_to_y_idx = []
    map_k_to_parent_sig = []

    # Infer target type from y (assumed consistent)
    target_type = type(y[0])
    unique_y = unique_labels(y)

    for cls_str in decoded_classes:
        if len(parent_indices) > 0:
            parts = cls_str.split("|")
            original_label = target_type(parts[0])
            parent_sig = "|".join(parts[1:])
        else:
            original_label = cls_str
            parent_sig = ""

        # Find index in unique_y
        y_idx = np.where(unique_y == original_label)[0][0]

        map_k_to_y_idx.append(y_idx)
        map_k_to_parent_sig.append(parent_sig)

    return {
        "parent_indices": parent_indices,
        "child_indices": child_indices,
        "estimator": sub_model,
        "map_y": np.array(map_k_to_y_idx),
        "map_parents": np.array(map_k_to_parent_sig),
    }


class _BaseAnDE(ClassifierMixin, BaseEstimator):
    """
    Base class for the AnDE family of algorithms.

    This class implements the **Generative Phase** using the **"Super-Class" (or Augmented Class)
    fitting strategy**. It serves as the foundation for both AnDE (Arithmetic Mean)
    and ALR/AnJE (Geometric Mean).

    **Mathematical Formulation:**

    An SPnDE (Super-Parent n-Dependence Estimator) models the joint probability
    $P(y, \\mathbf{x})$ by conditioning all attributes on the class $y$ and a subset
    of parent attributes $\\mathbf{x}_p$ (where $|\\mathbf{x}_p| = n$).

    To support mixed data types without re-implementing complex conditional distributions,
    we use the equivalence:

    .. math::
        P(y, \\mathbf{x}_p, \\mathbf{x}_{child}) = P(Y^*) \\prod P(x_i \\mid Y^*)

    Where $Y^* = (y, \\mathbf{x}_p)$ is the "Augmented Super-Class".

    Parameters
    ----------
    n_dependence : int, default=1
        The order of dependence 'n'.
        - n=0: Equivalent to Naive Bayes.
        - n=1: AODE (Averaged One-Dependence Estimators).
        - n=2: A2DE.

    n_bins : int, default=5
        Number of bins for discretizing numerical features ONLY when they act as super-parents.
        Children features remain continuous and are modeled by Gaussian distributions.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for discretization of super-parents.

    alpha : float, default=1.0
        Smoothing parameter passed to the internal MixedNB estimators.

    n_jobs : int, default=None
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    """

    def __init__(
        self,
        n_dependence=1,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        n_jobs=None,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
    ):
        self.n_dependence = n_dependence
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features

    def fit(self, X, y):
        """
        Generative fitting.
        Learns the joint probability P(y, x) for each subspace (SPODE) by counting frequencies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        # n_features_in_ is set by validate_data

        # --- 1. Discretize Parents ---
        self._discretizers_list = {}
        self._parent_data = np.zeros(X.shape, dtype=int)

        kwargs_discretizer = {"subsample": 200_000}

        if self.strategy == "quantile":
            # quantile_method was added in sklearn 1.7
            import sklearn

            sklearn_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
            if sklearn_version >= (1, 7):
                kwargs_discretizer["quantile_method"] = "linear"

        for i in range(self.n_features_in_):
            col = X[:, i]
            is_continuous = False
            if np.issubdtype(col.dtype, np.floating):
                if not np.all(np.mod(col, 1) == 0):
                    is_continuous = True

            if is_continuous:
                est = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode="ordinal",
                    strategy=self.strategy,
                    **kwargs_discretizer,
                )
                self._parent_data[:, i] = est.fit_transform(
                    col.reshape(-1, 1)
                ).flatten()
                self._discretizers_list[i] = est
            else:
                le = LabelEncoder()
                self._parent_data[:, i] = le.fit_transform(col)

        # --- 2. Build Ensemble (Parallelized) ---
        parent_combinations = list(
            combinations(range(self.n_features_in_), self.n_dependence)
        )
        if self.n_dependence == 0:
            parent_combinations = [()]

        # Use joblib to fit SPODEs in parallel
        self.ensemble_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_spode)(
                p_idx,
                X,
                y,
                self._parent_data,
                self.n_features_in_,
                self.alpha,
                self.categorical_features,
                self.bernoulli_features,
                self.gaussian_features,
            )
            for p_idx in parent_combinations
        )

        return self

    def _get_jll_per_model(self, X):
        """
        Computes the log-probability P(y, x | m) for each model 'm' in the ensemble.

        Returns
        -------
        jll_tensor : ndarray of shape (n_samples, n_classes, n_models)
            Contains log P(y, x) according to each SPODE.
            If a SPODE doesn't cover a sample (mismatch parents), returns -inf.

        X_parents_disc : ndarray
            The discretized version of X used for parent lookup (useful for Hybrid weights).
        """
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        X = validate_data(self, X, reset=False)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_models = len(self.ensemble_)

        # Discretize parents for test set
        X_parents_disc = X.copy()
        if hasattr(self, "_discretizers_list"):
            for col_idx, est in self._discretizers_list.items():
                X_parents_disc[:, col_idx] = est.transform(X[:, [col_idx]]).flatten()
        X_parents_disc = X_parents_disc.astype(int)

        # Output tensor initialized to a very small log-probability (approx 0 prob)
        # We avoid -inf so that Geometric mean (AnJE) doesn't collapse to 0 if one model misses.
        # -700 is close to the limit of exp() in float64 (~1e-304)
        jll_tensor = np.full((n_samples, n_classes, n_models), -700.0)

        # Sequential loop (Safer and easier to debug than Parallel for inference)
        for m_idx, model_info in enumerate(self.ensemble_):
            parent_indices = model_info["parent_indices"]
            estimator = model_info["estimator"]
            child_indices = model_info["child_indices"]
            map_y = model_info["map_y"]
            map_p = model_info["map_parents"]

            # 1. Get raw log probabilities for Y*
            if not child_indices:
                X_test_sub = np.zeros((n_samples, 1))
            else:
                X_test_sub = X[:, child_indices]

            jll_augmented = estimator._joint_log_likelihood(X_test_sub)

            if self.n_dependence == 0:
                jll_tensor[:, :, m_idx] = jll_augmented
                continue

            # 2. Filter based on parents
            test_parents_vals = X_parents_disc[:, parent_indices]
            test_signatures = np.array(
                ["|".join(map(str, row)) for row in test_parents_vals]
            )

            for k in range(len(map_p)):
                target_class_idx = map_y[k]
                required_sig = map_p[k]
                valid_mask = test_signatures == required_sig

                if np.any(valid_mask):
                    jll_tensor[valid_mask, target_class_idx, m_idx] = jll_augmented[
                        valid_mask, k
                    ]

        return jll_tensor, X_parents_disc


# =============================================================================
# 1. Generative Families (Classic AnDE)
# =============================================================================


class AnDE(_BaseAnDE):
    """
    Averaged n-Dependence Estimators (AnDE) [Generative].

    This is the standard generative model described by Webb et al. [1].
    It aggregates the predictions of sub-models (SPODEs) using an **Arithmetic Mean**.

    .. math::
        P(y|x) \\propto \\sum_{i} P_i(y, x) \\equiv \\log \\sum_{i} \\exp(\\text{JLL}_i)

    This implementation extends the original AnDE by supporting **mixed data types**
    (Gaussian/Categorical) through the Super-Class strategy.

    Parameters
    ----------
    n_dependence : int, default=1
        The order of dependence.
        - n=1: AODE (Averaged One-Dependence Estimators).
        - n=2: A2DE.

    n_bins : int, default=5
        Bins for discretizing super-parents.

    strategy : str, default='quantile'
        Discretization strategy.

    alpha : float, default=1.0
        Smoothing parameter.
    """

    def predict_log_proba(self, X):
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        jll_models, _ = self._get_jll_per_model(X)

        # Arithmetic Mean in Log Space: log(mean(exp(jll)))
        # = logsumexp(jll) - log(M)
        total_jll = logsumexp(jll_models, axis=2) - np.log(jll_models.shape[2])

        # Normalize to posterior P(y|x)
        log_prob_x = logsumexp(total_jll, axis=1)
        with np.errstate(invalid="ignore"):
            log_prob = total_jll - log_prob_x[:, np.newaxis]

        # Handle cases where all probs are 0 (log prob is -inf - -inf = nan)
        # We assign uniform probability or prior if everything is impossible
        mask_nan = np.isnan(log_prob)
        if np.any(mask_nan):
            # Fallback to uniform
            n_classes = len(self.classes_)
            log_prob[mask_nan] = -np.log(n_classes)

        return log_prob

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class AnJE(_BaseAnDE):
    """
    Averaged n-Join Estimators (AnJE) [Generative].

    A generative model similar to AnDE, but aggregates using a **Geometric Mean**.

    .. math::
        P(y|x) \\propto \\prod_{i} P_i(y, x) \\equiv \\sum_{i} \\log P_i(y, x)

    This model corresponds to the generative counterpart of ALR described by
    Zaidi et al. [2]. While often less accurate than AnDE on its own due to
    higher bias, it serves as the initialization basis for convex discriminative learning.

    Parameters
    ----------
    n_dependence : int, default=1
        The order of dependence.

    n_bins : int, default=5
        Bins for discretizing super-parents.

    strategy : str, default='quantile'
        Discretization strategy.

    alpha : float, default=1.0
        Smoothing parameter.
    """

    def predict_log_proba(self, X):
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        jll_models, _ = self._get_jll_per_model(X)

        # Geometric Mean in Log Space = Sum of logs
        total_jll = np.sum(jll_models, axis=2)

        # Normalize
        log_prob_x = logsumexp(total_jll, axis=1)
        with np.errstate(invalid="ignore"):
            log_prob = total_jll - log_prob_x[:, np.newaxis]

        mask_nan = np.isnan(log_prob)
        if np.any(mask_nan):
            # Fallback to uniform
            n_classes = len(self.classes_)
            log_prob[mask_nan] = -np.log(n_classes)

        return log_prob

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# =============================================================================
# 2. Discriminative / Hybrid Families (Learned Weights)
# =============================================================================


class _HybridOptimizer(_BaseAnDE):
    """
    Mixin implementing the 4 levels of parameter granularity for ALR/WeightedAnDE.

    This class handles the "Pre-conditioning" (generative fit) and the setup
    of the weight optimization problem.

    Reference: Zaidi et al. (2017), Section 5.4 [2].
    """

    def __init__(
        self,
        n_dependence=1,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        l2_reg=1e-4,
        max_iter=100,
        weight_level=1,
        n_jobs=None,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
    ):
        super().__init__(
            n_dependence,
            n_bins,
            strategy,
            alpha,
            n_jobs,
            categorical_features,
            bernoulli_features,
            gaussian_features,
        )
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.weight_level = weight_level

    def _setup_weights(self, X_parents_disc):
        """
        Prepares weight offsets based on granularity level.
        Handles n-dependence > 1 by linearizing parent value combinations.
        """
        n_models = len(self.ensemble_)
        n_classes = len(self.classes_)

        self._weight_offsets = [0]
        current_offset = 0

        # Store strides/cardinalities to re-use during test time if needed
        self._model_strides = []

        # 1. Determine Size of Weight Vector per Model
        for m_idx in range(n_models):
            p_indices = self.ensemble_[m_idx]["parent_indices"]

            # Cardinality for each parent feature (based on training data)
            # We use max()+1 to determine the range size 0..K
            # Shape: (n_parents,) e.g. [5, 5] for n=2 with 5 bins
            if len(p_indices) > 0:
                cards = np.max(X_parents_disc[:, p_indices], axis=0) + 1
            else:
                cards = np.array([1])  # Case n=0

            # Calculate total combinations (Cartesian product size)
            # n=1: size = cards[0]
            # n=2: size = cards[0] * cards[1]
            n_combinations = np.prod(cards)

            # Save cardinalities/strides for linear indexing
            # Standard C-order flattening: idx = v1*stride1 + v2*stride2 ... + vn*1
            # Strides are cumprod of sizes in reverse
            if len(p_indices) > 0:
                strides = np.cumprod(np.concatenate(([1], cards[::-1][:-1])))[::-1]
            else:
                strides = np.array([0])

            self._model_strides.append((p_indices, cards, strides))

            # Calculate final block size based on Level
            if self.weight_level == 1:  # Model
                size = 1
            elif self.weight_level == 2:  # Value
                size = n_combinations
            elif self.weight_level == 3:  # Class
                size = n_classes
            elif self.weight_level == 4:  # Value + Class
                size = n_combinations * n_classes

            current_offset += size
            self._weight_offsets.append(current_offset)

        self.n_weights_ = current_offset

        # 2. Pre-compute Weight Indices for Samples (Linearized)
        n_samples = X_parents_disc.shape[0]
        self._w_indices = np.zeros((n_samples, n_models), dtype=int)

        for m_idx in range(n_models):
            base_off = self._weight_offsets[m_idx]

            if self.weight_level in [1, 3]:
                self._w_indices[:, m_idx] = base_off
            elif self.weight_level in [2, 4]:
                p_indices, cards, strides = self._model_strides[m_idx]

                if len(p_indices) > 0:
                    # Get values for all parents: (N, n_parents)
                    vals = X_parents_disc[:, p_indices]

                    # Safety clip (critical for Test data robustness)
                    # Clip each column j to [0, cards[j]-1]
                    # This maps unseen outliers to the last valid bin instead of crashing
                    vals = np.minimum(vals, cards - 1)
                    vals = np.maximum(vals, 0)

                    # Linearize indices: dot product with strides
                    # Row i: v_i1*s1 + v_i2*s2 ...
                    linear_vals = vals @ strides

                    if self.weight_level == 2:
                        self._w_indices[:, m_idx] = base_off + linear_vals
                    else:  # Level 4 (Value * n_classes + Class_Offset)
                        # _w_indices points to the start of the class block for this value
                        self._w_indices[:, m_idx] = base_off + (linear_vals * n_classes)
                else:
                    self._w_indices[:, m_idx] = base_off

    def _get_weights_for_samples(self, flat_weights, n_samples):
        """
        Expands flat weights to (N, C, M) tensor based on pre-computed indices.
        """
        n_classes = len(self.classes_)
        n_models = len(self.ensemble_)

        if self.weight_level in [1, 2]:
            # Weights are class-independent (broadcast over C)
            W_gathered = flat_weights[self._w_indices]
            return W_gathered[:, np.newaxis, :]

        elif self.weight_level in [3, 4]:
            # Weights are class-specific
            W_out = np.zeros((n_samples, n_classes, n_models))
            for c in range(n_classes):
                indices_c = self._w_indices + c
                W_out[:, c, :] = flat_weights[indices_c]
            return W_out

    def fit(self, X, y):
        # 1. Generative Phase (Pre-conditioning)
        super().fit(X, y)

        # 2. Setup Weighting Structure
        X_check = validate_data(self, X, reset=False)
        # Parallel inference for JLL tensor
        jll_tensor, X_parents_disc = self._get_jll_per_model(X_check)
        jll_tensor = np.clip(jll_tensor, -1e10, 700)

        self._setup_weights(X_parents_disc)

        lb = LabelBinarizer()
        y_ohe = lb.fit_transform(y)
        if len(self.classes_) == 2:
            y_ohe = np.hstack((1 - y_ohe, y_ohe))

        # 3. Optimization
        def objective(w_flat):
            # Expand weights to (N, C, M)
            W_tensor = self._get_weights_for_samples(w_flat, X.shape[0])

            # Hook for ALR vs WeightedAnDE
            final_jll = self._calculate_final_jll(jll_tensor, W_tensor)
            final_jll = np.clip(final_jll, -1e10, 700)

            # Loss
            lse = logsumexp(final_jll, axis=1)
            log_proba = final_jll - lse[:, np.newaxis]

            true_class_log_probs = log_proba[y_ohe.astype(bool)]
            nll = -np.sum(true_class_log_probs)

            # L2 Regularization around 1.0 as per Zaidi et al.
            reg = self.l2_reg * np.sum((w_flat - 1.0) ** 2)
            return nll + reg

        initial_weights = np.ones(self.n_weights_)
        if isinstance(self, WeightedAnDE):
            initial_weights /= len(self.ensemble_)

        bounds = [(0, None) for _ in range(self.n_weights_)]

        # Suppress warnings during optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                objective,
                initial_weights,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter},
            )

        self.learned_weights_ = res.x
        return self

    def predict_log_proba(self, X):
        jll_models, X_parents_disc = self._get_jll_per_model(X)
        jll_models = np.clip(jll_models, -1e10, 700)

        # Re-calc indices for Test data
        self._setup_weights(X_parents_disc)
        W_tensor = self._get_weights_for_samples(self.learned_weights_, X.shape[0])

        final_jll = self._calculate_final_jll(jll_models, W_tensor)
        final_jll = np.clip(final_jll, -1e10, 700)

        log_prob_x = logsumexp(final_jll, axis=1)
        with np.errstate(invalid="ignore"):
            log_prob = final_jll - log_prob_x[:, np.newaxis]

        mask_nan = np.isnan(log_prob)
        if np.any(mask_nan):
            n_classes = len(self.classes_)
            log_prob[mask_nan] = -np.log(n_classes)

        return log_prob

    def _calculate_final_jll(self, jll_tensor, W_tensor):
        raise NotImplementedError


class ALR(_HybridOptimizer, AnJE):
    """
    Accelerated Logistic Regression (ALR) [Hybrid].

    A hybrid generative-discriminative classifier.
    Optimizes weights in log-space (Convex).

    Supports 4 Levels of Weight Granularity:
    1. Per Model (Default)
    2. Per Parent Value
    3. Per Class
    4. Per Parent Value & Class

    Parameters
    ----------
    weight_level : int, default=1
        Granularity of weights (1-4).
    l2_reg : float, default=1e-4
        L2 regularization.
    """

    def _calculate_final_jll(self, jll_tensor, W_tensor):
        # Geometric Mean Logic: Sum(w * logP)
        weighted_jll = jll_tensor * W_tensor
        weighted_jll = np.nan_to_num(weighted_jll, nan=0.0)
        return np.sum(weighted_jll, axis=2)


class WeightedAnDE(_HybridOptimizer, AnDE):
    """
    Weighted AnDE [Hybrid].

    A discriminative weighting of standard AnDE (Arithmetic Mean).
    Optimization is Non-Convex.
    """

    def _calculate_final_jll(self, jll_tensor, W_tensor):
        # Arithmetic Mean Logic: LogSumExp(log(w) + logP)
        w_safe = np.maximum(W_tensor, 1e-10)
        weighted_log_terms = np.log(w_safe) + jll_tensor
        weighted_log_terms = np.clip(weighted_log_terms, -700, 700)
        return logsumexp(weighted_log_terms, axis=2)
