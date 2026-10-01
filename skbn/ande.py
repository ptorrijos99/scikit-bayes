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
.. [2] Webb, G. I., Boughton, J., Zheng, F., Ting, K. M., & Salem, H. (2011).
       Learning by extrapolation from marginal to full-multivariate probability
       distributions: Decreasingly naive Bayesian classification. Machine Learning, 86(2), 233-272.
.. [3] Zaidi, N. A., Webb, G. I., Carman, M. J., & Petitjean, F. (2017).
       Efficient parameter learning of Bayesian network classifiers.
       Machine Learning, 106(9-10), 1289-1329.
.. [4] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., & Cerquides, J. (2016).
       ALR^n: Accelerated higher-order logistic regression. Machine Learning, 104(2-3), 151-194.
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
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    column_or_1d,
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
    parent_cards,
    categorical_features=None,
    bernoulli_features=None,
    gaussian_features=None,
    var_prior_weight=0.0,
):
    """Fits a single SPODE (Sub-model) in parallel.

    Uses integer-based Y* encoding for speed and robustness.
    Y* = class_idx * n_parent_combos + parent_linear_index

    The augmented-class prior P(Y*) is Laplace-smoothed over the *full*
    grid of augmented states (classes x parent combinations), so that
    states never observed in training keep a finite, smoothed probability.
    """
    unique_y = np.unique(y)
    y_int = np.searchsorted(unique_y, y)
    n_classes = len(unique_y)

    # A. Compute integer Y*
    if len(parent_indices) > 0:
        p_vals = parent_data[:, parent_indices]
        cards = parent_cards[list(parent_indices)]
        strides = np.cumprod(np.concatenate(([1], cards[::-1][:-1])))[::-1]
        n_combos = int(np.prod(cards))
        parent_keys = (p_vals * strides).sum(axis=1).astype(np.int64)
        y_star = y_int.astype(np.int64) * n_combos + parent_keys
    else:
        cards = np.array([1])
        strides = np.array([0])
        n_combos = 1
        parent_keys = np.zeros(len(y), dtype=np.int64)
        y_star = y_int.astype(np.int64)

    # B. Identify child features
    child_indices = [i for i in range(n_features) if i not in parent_indices]

    if not child_indices:
        X_train_sub = np.zeros((X.shape[0], 1))
    else:
        X_train_sub = X[:, child_indices]

    # Map global feature indices to local sub-indices
    global_to_sub = {g_idx: s_idx for s_idx, g_idx in enumerate(child_indices)}

    def _to_sub(features):
        if features is None:
            return None
        sub = [global_to_sub[g] for g in features if g in global_to_sub]
        return sub if sub else None

    # C. Fit MixedNB on (X_children, Y*). With no parents (n=0) the component
    #    is plain Naive Bayes and is kept identical to it (no shrinkage, MLE prior).
    has_parents = len(parent_indices) > 0
    sub_model = MixedNB(
        alpha=alpha,
        categorical_features=_to_sub(categorical_features),
        bernoulli_features=_to_sub(bernoulli_features),
        gaussian_features=_to_sub(gaussian_features),
        var_prior_weight=var_prior_weight if has_parents else 0.0,
    )
    sub_model.fit(X_train_sub, y_star)

    # D. Laplace-smoothed augmented prior over the full grid of Y* states.
    n_total = len(y)
    n_states = n_classes * n_combos
    if has_parents:
        counts = np.bincount(np.searchsorted(sub_model.classes_, y_star))
        denom = n_total + alpha * n_states
        sub_model.class_log_prior_ = np.log((counts + alpha) / denom)
        log_prior_unseen = np.log(alpha / denom) if alpha > 0 else -np.inf
    else:
        log_prior_unseen = -np.inf

    # E. Observed parent instantiations and their frequencies (for the
    #    minimum-frequency rule that decides whether a SPODE is used).
    obs_parent_keys, obs_parent_counts = np.unique(parent_keys, return_counts=True)

    return {
        "parent_indices": parent_indices,
        "child_indices": child_indices,
        "estimator": sub_model,
        "ystar_keys": np.asarray(sub_model.classes_, dtype=np.int64),
        "n_parent_combos": n_combos,
        "parent_cards": cards,
        "parent_strides": strides,
        "obs_parent_keys": obs_parent_keys,
        "obs_parent_counts": obs_parent_counts,
        "log_prior_unseen": log_prior_unseen,
    }


def _lookup(sorted_keys, queries):
    """Positions of ``queries`` in ``sorted_keys`` and a mask of hits."""
    pos = np.searchsorted(sorted_keys, queries)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    return pos, sorted_keys[pos] == queries


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

    **Zero counts.** (i) As in the smoothed estimates of AnDE/AnJE, the
    augmented prior $P(Y^*)$ is Laplace-smoothed over every augmented state;
    (ii) when a test instance falls in an augmented state $(y, \\mathbf{x}_p)$
    that never occurred in training, the child conditionals are taken from
    ``unseen_state``: ``"backoff"`` uses the class-level estimates
    $P(x_i \\mid y)$, ``"laplace"`` the uninformed estimate that Laplace
    smoothing gives an empty cell (uniform for discrete children, the pooled
    Gaussian of the attribute for continuous ones); and (iii) as in AODE, a
    SPnDE whose parent instantiation $\\mathbf{x}_p$ occurred fewer than
    ``min_parent_count`` times is excluded from the ensemble for that instance.
    If no SPnDE qualifies, the prediction falls back to Naive Bayes.

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
        Smoothing parameter passed to the internal MixedNB estimators and used
        for the augmented-class prior.

    n_jobs : int, default=None
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    var_prior_weight : float, default=1.0
        Pseudo-observations shrinking each Gaussian child variance towards the
        pooled variance of the feature (see :class:`MixedNB`).

    min_parent_count : int, default=1
        Minimum training frequency of a parent instantiation for its SPnDE to
        take part in the prediction of an instance.

    unseen_state : {"laplace", "backoff"}, default="laplace"
        Child conditionals used for an augmented state never seen in training.
        ``"laplace"`` is the standard smoothed estimate of an empty cell;
        ``"backoff"`` reuses the class-level conditionals. An ablation over the
        benchmark found no practical difference between them.
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
        var_prior_weight=1.0,
        min_parent_count=1,
        unseen_state="laplace",
    ):
        self.n_dependence = n_dependence
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features
        self.var_prior_weight = var_prior_weight
        self.min_parent_count = min_parent_count
        self.unseen_state = unseen_state

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
        self._discretizers_list = {}  # KBinsDiscretizer or LabelEncoder per feature
        self._parent_data = np.zeros(X.shape, dtype=int)

        # Use all data for discretizer to ensure maximum precision
        kwargs_discretizer = {"subsample": None}

        if self.strategy == "quantile":
            # quantile_method was added in sklearn 1.7
            import sklearn

            sklearn_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
            if sklearn_version >= (1, 7):
                kwargs_discretizer["quantile_method"] = "linear"

        # Declared feature types take precedence over the dtype heuristic, so
        # integer-valued continuous attributes (ages, amounts, counts) are
        # binned instead of being used as high-cardinality categorical parents.
        declared_gaussian = {int(i) for i in self.gaussian_features or []}
        declared_discrete = {
            int(i)
            for i in list(self.categorical_features or [])
            + list(self.bernoulli_features or [])
        }

        for i in range(self.n_features_in_):
            col = X[:, i]
            if i in declared_gaussian:
                is_continuous = True
            elif i in declared_discrete:
                is_continuous = False
            else:
                is_continuous = np.issubdtype(col.dtype, np.floating) and not np.all(
                    np.mod(col, 1) == 0
                )

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
                # Save LabelEncoder so we can transform test data correctly
                le = LabelEncoder()
                self._parent_data[:, i] = le.fit_transform(col)
                self._discretizers_list[i] = le

        # Compute per-feature cardinalities from training data
        self._parent_cards = np.max(self._parent_data, axis=0).astype(int) + 1

        # --- 2. Class-level back-off model (Naive Bayes on the original class) ---
        self.backoff_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=self.gaussian_features,
            var_prior_weight=self.var_prior_weight,
        ).fit(X, y)
        if self.unseen_state not in ("backoff", "laplace"):
            raise ValueError(
                "unseen_state must be 'backoff' or 'laplace', got"
                f" {self.unseen_state!r}"
            )
        gauss = self.backoff_.feature_types_["gaussian"]
        Xg = np.asarray(X[:, gauss], dtype=float) if gauss else np.zeros((len(X), 0))
        self._pooled_mean = Xg.mean(axis=0) if gauss else np.zeros(0)
        self._pooled_var = Xg.var(axis=0) + (
            self.backoff_.estimators_["gaussian"].epsilon_ if gauss else 0.0
        )

        # --- 3. Build Ensemble (Parallelized) ---
        parent_combinations = list(
            combinations(range(self.n_features_in_), self.n_dependence)
        )
        if self.n_dependence == 0:
            parent_combinations = [()]

        # Use joblib to fit SPODEs in parallel
        # Disabling memory mapping (max_nbytes=None, mmap_mode=None) prevents joblib from writing temporary files to disk.
        self.ensemble_ = Parallel(
            n_jobs=self.n_jobs,
            max_nbytes=None,
            mmap_mode=None,
        )(
            delayed(_fit_spode)(
                p_idx,
                X,
                y,
                self._parent_data,
                self.n_features_in_,
                self.alpha,
                self._parent_cards,
                self.categorical_features,
                self.bernoulli_features,
                self.gaussian_features,
                self.var_prior_weight,
            )
            for p_idx in parent_combinations
        )

        return self

    def _discretize_parents(self, X):
        """Maps (validated) X to the integer parent codes used at fit time.

        Returns the codes (clipped to the training range) and a boolean mask
        of categorical values never seen in training; a component whose parent
        value is unseen is excluded for that sample (minimum-frequency rule).
        """
        n_samples = X.shape[0]
        X_parents_disc = np.zeros((n_samples, self.n_features_in_), dtype=int)
        unseen = np.zeros((n_samples, self.n_features_in_), dtype=bool)
        for col_idx in range(self.n_features_in_):
            if col_idx in self._discretizers_list:
                est = self._discretizers_list[col_idx]
                if isinstance(est, LabelEncoder):
                    col = X[:, col_idx]
                    known_mask = np.isin(col, est.classes_)
                    unseen[:, col_idx] = ~known_mask
                    safe_col = np.where(known_mask, col, est.classes_[0])
                    X_parents_disc[:, col_idx] = est.transform(safe_col)
                else:
                    X_parents_disc[:, col_idx] = (
                        est.transform(X[:, [col_idx]]).flatten().astype(int)
                    )
            else:
                X_parents_disc[:, col_idx] = X[:, col_idx].astype(int)

        # Clip to training cardinalities (safety for unseen test values)
        np.clip(X_parents_disc, 0, self._parent_cards - 1, out=X_parents_disc)
        return X_parents_disc, unseen

    def _uninformed_log_likelihood(self, X):
        """Class-independent log-likelihood of each feature for an empty cell.

        Laplace smoothing of an empty cell gives a uniform conditional for a
        discrete child; for a Gaussian child the uninformed estimate is the
        pooled Gaussian of the attribute. Returns shape (n_samples, n_features).
        """
        out = np.zeros((X.shape[0], self.n_features_in_))
        types = self.backoff_.feature_types_
        if types["gaussian"]:
            g = types["gaussian"]
            diff = np.asarray(X[:, g], dtype=float) - self._pooled_mean
            out[:, g] = -0.5 * (
                diff**2 / self._pooled_var + np.log(2.0 * np.pi * self._pooled_var)
            )
        if types["categorical"] and "categorical" in self.backoff_.estimators_:
            out[:, types["categorical"]] = -np.log(self.backoff_._cat_cardinalities)
        if types["bernoulli"]:
            out[:, types["bernoulli"]] = -np.log(2.0)
        return out

    def _get_jll_per_model(self, X):
        """
        Computes the log-probability log P_m(y, x) for each model 'm' in the ensemble.

        Returns
        -------
        jll_tensor : ndarray of shape (n_samples, n_classes, n_models)
            Contains log P(y, x) according to each SPODE.

        X_parents_disc : ndarray of shape (n_samples, n_features)
            The discretized version of X used for parent lookup (useful for Hybrid weights).

        active : ndarray of bool, shape (n_samples, n_models)
            Whether SPODE ``m`` takes part in the prediction of each sample
            (its parent instantiation occurred at least ``min_parent_count``
            times in training).

        nb_jll : ndarray of shape (n_samples, n_classes)
            Naive Bayes joint log-likelihood, used for samples with no active
            SPODE. Only rows where ``active.any(axis=1)`` is False are filled.
        """
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        X = validate_data(self, X, reset=False)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_models = len(self.ensemble_)

        X_parents_disc, unseen = self._discretize_parents(X)

        # --- Pass 1: parent keys, activity and augmented-state hits ---
        keys, hits, positions = [], [], []
        active = np.ones((n_samples, n_models), dtype=bool)
        needs_backoff = np.zeros(n_samples, dtype=bool)
        class_offsets = np.arange(n_classes, dtype=np.int64)
        for m_idx, info in enumerate(self.ensemble_):
            if self.n_dependence == 0:
                keys.append(None)
                hits.append(None)
                positions.append(None)
                continue
            p_vals = X_parents_disc[:, info["parent_indices"]]
            pk = (p_vals * info["parent_strides"]).sum(axis=1).astype(np.int64)
            pos_p, found_p = _lookup(info["obs_parent_keys"], pk)
            counts = np.where(found_p, info["obs_parent_counts"][pos_p], 0)
            active[:, m_idx] = (counts >= max(self.min_parent_count, 1)) & ~unseen[
                :, info["parent_indices"]
            ].any(axis=1)

            queries = (
                class_offsets[np.newaxis, :] * info["n_parent_combos"] + pk[:, None]
            )
            pos, hit = _lookup(info["ystar_keys"], queries)
            keys.append(pk)
            hits.append(hit)
            positions.append(pos)
            needs_backoff |= active[:, m_idx] & ~hit.all(axis=1)

        no_active = ~active.any(axis=1)
        rows_ll = np.flatnonzero(needs_backoff | no_active)
        row_map = np.full(n_samples, -1, dtype=np.int64)
        row_map[rows_ll] = np.arange(len(rows_ll))
        feat_ll = (
            self.backoff_._feature_log_likelihood(X[rows_ll])
            if len(rows_ll)
            else np.zeros((0, n_classes, self.n_features_in_))
        )

        nb_jll = np.zeros((n_samples, n_classes))
        if no_active.any():
            r = row_map[no_active]
            nb_jll[no_active] = feat_ll[r].sum(axis=2) + self.backoff_.class_log_prior_

        # --- Pass 2: fill the tensor ---
        jll_tensor = np.zeros((n_samples, n_classes, n_models))
        for m_idx, info in enumerate(self.ensemble_):
            child_indices = info["child_indices"]
            X_sub = X[:, child_indices] if child_indices else np.zeros((n_samples, 1))
            jll_aug = info["estimator"]._joint_log_likelihood(X_sub)

            if self.n_dependence == 0:
                jll_tensor[:, :, m_idx] = jll_aug
                continue

            hit, pos = hits[m_idx], positions[m_idx]
            rows = np.arange(n_samples)[:, None]
            jll_tensor[:, :, m_idx] = np.where(hit, jll_aug[rows, pos], 0.0)

            miss = active[:, m_idx][:, None] & ~hit
            if miss.any():
                r_idx, c_idx = np.nonzero(miss)
                if getattr(self, "unseen_state", "laplace") == "laplace":
                    uninf = self._uninformed_log_likelihood(X[r_idx][:, :])
                    child_ll = uninf[:, child_indices].sum(axis=1)
                else:
                    child_ll = feat_ll[row_map[r_idx], c_idx][:, child_indices].sum(
                        axis=1
                    )
                jll_tensor[r_idx, c_idx, m_idx] = info["log_prior_unseen"] + child_ll

        return jll_tensor, X_parents_disc, active, nb_jll

    @staticmethod
    def _normalize(scores):
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        return np.nan_to_num(log_prob, nan=-np.log(scores.shape[1]))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# =============================================================================
# 1. Generative Families (Classic AnDE)
# =============================================================================


def _arithmetic_scores(jll, active, nb_jll):
    """log of the mean of the active components' joints (per sample)."""
    n_active = active.sum(axis=1)
    masked = np.where(active[:, None, :], jll, -np.inf)
    with np.errstate(divide="ignore"):
        scores = logsumexp(masked, axis=2) - np.log(np.maximum(n_active, 1))[:, None]
    fallback = n_active == 0
    scores[fallback] = nb_jll[fallback]
    return scores


def _geometric_scores(jll, active, nb_jll):
    """Mean of the active components' log-joints (log geometric mean)."""
    n_active = active.sum(axis=1)
    scores = np.where(active[:, None, :], jll, 0.0).sum(axis=2)
    scores /= np.maximum(n_active, 1)[:, None]
    fallback = n_active == 0
    scores[fallback] = nb_jll[fallback]
    return scores


class AnDE(_BaseAnDE):
    """
    Averaged n-Dependence Estimators (AnDE) [Generative].

    This is the standard generative model described by Webb et al. [1].
    It aggregates the predictions of sub-models (SPODEs) using an **Arithmetic Mean**
    of their joint probabilities.

    .. math::
        P(y|x) \\propto \\frac{1}{|M|} \\sum_{i} P_i(y, x)

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
        jll, _, active, nb_jll = self._get_jll_per_model(X)
        return self._normalize(_arithmetic_scores(jll, active, nb_jll))


class AnJE(_BaseAnDE):
    """
    Averaged n-Join Estimators (AnJE) [Generative].

    A generative model similar to AnDE, but aggregates using a **Geometric Mean**
    of the components' joint probabilities.

    .. math::
        P(y|x) \\propto \\prod_{i} P_i(y, x)^{1/|M|}

    This model corresponds to the generative counterpart of ALR described by
    Zaidi et al. [4]. While often less accurate than AnDE on its own due to
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
        jll, _, active, nb_jll = self._get_jll_per_model(X)
        return self._normalize(_geometric_scores(jll, active, nb_jll))


# =============================================================================
# 2. Discriminative / Hybrid Families (Learned Weights)
# =============================================================================


class _HybridOptimizer(_BaseAnDE):
    """
    Mixin implementing the 4 levels of parameter granularity for ALR/WeightedAnDE.

    This class handles the "Pre-conditioning" (generative fit) and the setup
    of the weight optimization problem. All objectives are optimized with
    L-BFGS-B using exact (analytic) gradients.

    Weight levels (per SPnDE component ``m``):

    1. one weight per component;
    2. one weight per parent instantiation of the component;
    3. one weight per class of the component;
    4. one weight per augmented state (class x parent instantiation).

    Reference: Zaidi et al. (2017), Section 5.4 [3].
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
        modular=False,
        var_prior_weight=1.0,
        min_parent_count=1,
        unseen_state="laplace",
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
            var_prior_weight,
            min_parent_count,
            unseen_state,
        )
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.weight_level = weight_level
        self.modular = modular

    # --- Subclass hooks -----------------------------------------------------
    _log_space = False  # WeightedAnDE optimizes log-weights

    def _scores_and_grad_factor(self, theta_samples, jll, active):
        """Return ``(scores, dscore/dtheta)`` for per-sample expanded parameters.

        ``theta_samples`` has shape (N, C, M) (or (N, 1, M) for class-independent
        levels). The derivative has shape (N, C, M).
        """
        raise NotImplementedError

    # --- Weight bookkeeping -------------------------------------------------
    @property
    def _class_specific(self):
        return self.weight_level in (3, 4)

    def _setup_weights(self, X_parents_disc):
        """
        Prepares weight offsets based on granularity level.
        Uses training-time cardinalities stored in the ensemble for consistency.
        """
        if self.weight_level not in (1, 2, 3, 4):
            raise ValueError(f"weight_level must be in 1..4, got {self.weight_level}")
        n_classes = len(self.classes_)
        self._weight_offsets = [0]
        for info in self.ensemble_:
            n_comb = info["n_parent_combos"]
            size = {1: 1, 2: n_comb, 3: n_classes, 4: n_comb * n_classes}[
                self.weight_level
            ]
            self._weight_offsets.append(self._weight_offsets[-1] + size)
        self.n_weights_ = self._weight_offsets[-1]
        self._w_indices = self._compute_base_indices(X_parents_disc)

    def _compute_base_indices(self, X_parents_disc):
        """Index of the first weight used by each (sample, component)."""
        n_samples = X_parents_disc.shape[0]
        n_classes = len(self.classes_)
        base = np.zeros((n_samples, len(self.ensemble_)), dtype=np.int64)
        for m_idx, info in enumerate(self.ensemble_):
            off = self._weight_offsets[m_idx]
            if self.weight_level in (1, 3) or len(info["parent_indices"]) == 0:
                base[:, m_idx] = off
                continue
            vals = X_parents_disc[:, info["parent_indices"]]
            vals = np.clip(vals, 0, info["parent_cards"] - 1)
            linear = vals @ info["parent_strides"]
            base[:, m_idx] = off + (
                linear * n_classes if self.weight_level == 4 else linear
            )
        return base

    def _expand_indices(self, base, n_classes):
        if self._class_specific:
            return base[:, None, :] + np.arange(n_classes)[None, :, None]
        return base[:, None, :]

    # --- Objective ----------------------------------------------------------
    def _to_weights(self, theta):
        return np.exp(theta) if self._log_space else theta

    def _objective(self, theta, jll, active, idx, Y):
        w = self._to_weights(theta)
        scores, dscore = self._scores_and_grad_factor(theta[idx], jll, active)
        log_norm = logsumexp(scores, axis=1, keepdims=True)
        nll = -np.sum((scores - log_norm) * Y)
        resid = np.exp(scores - log_norm) - Y  # dNLL/dscores

        g_full = resid[:, :, None] * dscore
        if idx.shape[1] == 1:  # class-independent weights: sum over classes
            g_full = g_full.sum(axis=1, keepdims=True)
        grad = np.bincount(
            np.broadcast_to(idx, g_full.shape).ravel(),
            weights=g_full.ravel(),
            minlength=len(theta),
        )

        reg = self.l2_reg * np.sum((w - 1.0) ** 2)
        dreg = 2.0 * self.l2_reg * (w - 1.0)
        if self._log_space:
            dreg = dreg * w
        return nll + reg, grad + dreg

    def _optimize(self, theta0, jll, active, idx, Y):
        bounds = None if self._log_space else [(0.0, None)] * len(theta0)
        res = minimize(
            self._objective,
            theta0,
            args=(jll, active, idx, Y),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        self.n_iter_ = getattr(self, "n_iter_", 0) + res.nit
        return res.x

    def fit(self, X, y):
        # 1. Generative Phase (Pre-conditioning)
        super().fit(X, y)

        # 2. Setup Weighting Structure
        X_check = validate_data(self, X, reset=False)
        jll, X_parents_disc, active, _ = self._get_jll_per_model(X_check)
        self._setup_weights(X_parents_disc)

        n_classes = len(self.classes_)
        # The generative fit has already validated y (and warned on a column vector)
        y_idx = np.searchsorted(self.classes_, column_or_1d(y))
        Y = np.eye(n_classes)[y_idx]

        # Samples with no active component are predicted by the NB fallback
        # and do not depend on the weights.
        keep = active.any(axis=1)
        jll, active, Y = jll[keep], active[keep], Y[keep]
        base = self._w_indices[keep]

        theta = (
            np.zeros(self.n_weights_) if self._log_space else np.ones(self.n_weights_)
        )
        self.n_iter_ = 0

        # 3. Optimization
        if self.modular:
            if self._log_space and not self._class_specific:
                warnings.warn(
                    (
                        "Modular WeightedAnDE with weight_level 1 or 2 is equivalent to"
                        " the generative AnDE: a class-independent weight cancels in"
                        " the per-component softmax, so it is not identifiable from"
                        " that component's conditional likelihood. Weights are fixed"
                        " to 1."
                    ),
                    UserWarning,
                )
            else:
                for m_idx in range(len(self.ensemble_)):
                    rows = active[:, m_idx]
                    if not rows.any():
                        continue
                    lo, hi = (
                        self._weight_offsets[m_idx],
                        self._weight_offsets[m_idx + 1],
                    )
                    local_idx = self._expand_indices(
                        base[rows, m_idx : m_idx + 1] - lo, n_classes
                    )
                    theta[lo:hi] = self._optimize(
                        theta[lo:hi],
                        jll[rows][:, :, m_idx : m_idx + 1],
                        np.ones((rows.sum(), 1), dtype=bool),
                        local_idx,
                        Y[rows],
                    )
        else:
            idx = self._expand_indices(base, n_classes)
            theta = self._optimize(theta, jll, active, idx, Y)

        self.coef_theta_ = theta
        self.learned_weights_ = self._to_weights(theta)
        return self

    def predict_log_proba(self, X):
        check_is_fitted(self, attributes=["classes_", "ensemble_", "coef_theta_"])
        jll, X_parents_disc, active, nb_jll = self._get_jll_per_model(X)
        base = self._compute_base_indices(X_parents_disc)
        idx = self._expand_indices(base, len(self.classes_))
        scores, _ = self._scores_and_grad_factor(self.coef_theta_[idx], jll, active)
        fallback = ~active.any(axis=1)
        scores[fallback] = nb_jll[fallback]
        return self._normalize(scores)


class ALR(_HybridOptimizer, AnJE):
    """
    Accelerated Logistic Regression (ALR) [Hybrid].

    A hybrid generative-discriminative classifier that combines the generative
    topology of Averaged n-Join Estimators (AnJE) with discriminative
    weight optimization. The score is the weighted log geometric mean

    .. math::
        s(y, x) = \\frac{1}{|M|} \\sum_m w_m(y, x) \\log P_m(y, x),

    which is linear in the weights, so the conditional log-likelihood is
    concave and solvable via L-BFGS-B. At ``w = 1`` the model is exactly AnJE.

    Supports 4 Levels of Weight Granularity (from coarsest to finest):
    1. Per Model (Default) - One weight per ensemble member.
    2. Per Parent Value - One weight per parent feature value.
    3. Per Class - One weight per target class per model.
    4. Per Parent Value & Class - One weight per parent value per class.

    Parameters
    ----------
    n_dependence : int, default=1
        The number of parent features conditioned upon (n-dependence).
        0 corresponds to Naive Bayes topology.
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter for probabilities.
    n_bins : int, default=5
        Maximum number of bins for discretization.
    weight_level : int, default=1
        Granularity of weights (1-4).
    l2_reg : float, default=1e-4
        L2 regularization towards the generative solution (w = 1).
    max_iter : int, default=100
        Maximum number of L-BFGS-B iterations.
    modular : bool, default=False
        If True, the weights of each component are fitted independently on
        that component's own conditional log-likelihood.
    n_jobs : int, default=None
        Number of parallel jobs to run during the generative phase.
    """

    _log_space = False

    def _scores_and_grad_factor(self, theta_samples, jll, active):
        n_active = np.maximum(active.sum(axis=1), 1)[:, None, None]
        jll_active = np.where(active[:, None, :], jll, 0.0) / n_active
        scores = np.sum(theta_samples * jll_active, axis=2)
        return scores, jll_active


class WeightedAnDE(_HybridOptimizer, AnDE):
    """
    Weighted Averaged n-Dependence Estimators (WeightedAnDE) [Hybrid].

    A discriminative weighting of the standard AnDE (Arithmetic Mean) ensemble:

    .. math::
        s(y, x) = \\log \\sum_m w_m(y, x) P_m(y, x).

    Weights are optimized in log-space (``w = exp(theta)``). Jointly fitting
    all components is non-convex; in ``modular`` mode each component is fitted
    on its own conditional log-likelihood, which is convex in ``theta``. In
    modular mode only class-specific levels (3 and 4) are identifiable; levels
    1 and 2 reduce to the generative AnDE.

    Supports 4 Levels of Weight Granularity (from coarsest to finest):
    1. Per Model (Default) - One weight per ensemble member.
    2. Per Parent Value - One weight per parent feature value.
    3. Per Class - One weight per target class per model.
    4. Per Parent Value & Class - One weight per parent value per class.

    Parameters
    ----------
    n_dependence : int, default=1
        The number of parent features conditioned upon (n-dependence).
        0 corresponds to Naive Bayes topology.
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter for probabilities.
    n_bins : int, default=5
        Maximum number of bins for discretization.
    weight_level : int, default=1
        Granularity of weights (1-4).
    l2_reg : float, default=1e-4
        L2 regularization towards the generative solution (w = 1).
    max_iter : int, default=100
        Maximum number of L-BFGS-B iterations.
    modular : bool, default=False
        If True, the weights of each component are fitted independently.
    n_jobs : int, default=None
        Number of parallel jobs to run during the generative phase.
    """

    _log_space = True

    def _scores_and_grad_factor(self, theta_samples, jll, active):
        terms = np.where(active[:, None, :], jll + theta_samples, -np.inf)
        scores = logsumexp(terms, axis=2)
        with np.errstate(invalid="ignore"):
            resp = np.exp(terms - scores[:, :, None])
        return scores, np.nan_to_num(resp)
