"""
Averaged n-Dependence Estimators (AnDE) and their discriminative weighting.

This module implements ensembles of SuperParent n-Dependence Estimators
(SPnDEs) that support mixed data types (continuous, categorical, binary)
natively through the "Super-Class" strategy:

1.  **AnDE:** the generative AODE/A2DE ensemble [1]_ [2]_, pooled with the
    arithmetic mean (the standard model) or the geometric mean.
2.  **WeightedAnDE:** discriminative weights learned on top of AnDE, either
    one per component of the ensemble or one per factor of each component.

The Averaged n-Join Estimators and ALR of Zaidi et al. [4]_ are built on
n-joins rather than SPnDEs and live in :mod:`skbn.anje`, together with their
extension to Gaussian children on the factors of the SPnDEs.

References
----------
.. [1] Webb, G. I., Boughton, J., & Wang, Z. (2005). Not so naive Bayes:
       Aggregating one-dependence estimators. Machine Learning, 58(1), 5-24.
.. [2] Webb, G. I., Boughton, J., Zheng, F., Ting, K. M., & Salem, H. (2011).
       Learning by extrapolation from marginal to full-multivariate probability
       distributions: Decreasingly naive Bayesian classification. Machine Learning, 86(2), 233-272.
.. [3] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
       Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
       Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.
.. [4] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., & Cerquides, J. (2016).
       ALR^n: Accelerated higher-order logistic regression. Machine Learning, 104(2-3), 151-194.
.. [5] Zheng, X., Lin, Z., Xu, H., Chen, C., & Ye, T. (2015). Efficient
       learning ensemble SuperParent-one-dependence estimator by maximizing
       conditional log likelihood. Expert Systems with Applications, 42,
       7732-7745.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import warnings
from itertools import combinations
from numbers import Integral, Real
from typing import NamedTuple, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    column_or_1d,
    validate_data,
)

from ._core import (
    _AttributeEncoder,
    _fit_weights,
    _row_chunks,
    _warn_not_converged,
)
from .mixed_nb import MixedNB

_LOG_TINY = np.log(np.finfo(float).tiny)

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
    n_classes=None,
    X_int=None,
    categories=None,
):
    """Fits a single SPODE (Sub-model) in parallel.

    Uses integer-based Y* encoding for speed and robustness.
    Y* = class_idx * n_parent_combos + parent_linear_index

    ``y`` holds class indices (0..n_classes-1), ``categories`` the sorted
    categories of every categorical column of ``X`` (None for the others) and
    ``X_int`` the codes of those columns; all are computed once by the
    ensemble rather than once per component.

    The augmented-class prior P(Y*) is Laplace-smoothed over the *full*
    grid of augmented states (classes x parent combinations), so that
    states never observed in training keep a finite, smoothed probability.
    """
    if n_classes is None:  # raw labels
        unique_y = np.unique(y)
        y_int = np.searchsorted(unique_y, y)
        n_classes = len(unique_y)
    else:
        y_int = y

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

    # The component's model works on the child columns of X, read in place
    if not child_indices:
        X_fit, X_int_fit, columns = np.zeros((X.shape[0], 1)), None, None
    else:
        X_fit, X_int_fit, columns = X, X_int, child_indices

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
    ystar_classes, ystar_idx = np.unique(y_star, return_inverse=True)
    sub_model._fit_encoded(
        X_fit,
        ystar_idx,
        ystar_classes,
        X_int=X_int_fit,
        columns=columns,
        categories=categories if columns is not None else None,
    )
    counts = np.bincount(ystar_idx, minlength=len(ystar_classes))

    # D. Laplace-smoothed augmented prior over the full grid of Y* states.
    n_total = len(y)
    n_states = n_classes * n_combos
    if has_parents:
        denom = n_total + alpha * n_states
        sub_model.class_log_prior_ = np.log((counts + alpha) / denom)
        log_prior_unseen = np.log(alpha / denom) if alpha > 0 else -np.inf
    else:
        log_prior_unseen = -np.inf

    # E. Observed parent instantiations and their frequencies (for the
    #    minimum-frequency rule that decides whether a SPODE is used),
    #    aggregated from the augmented states instead of re-sorting the data.
    obs_parent_keys, inv = np.unique(ystar_classes % n_combos, return_inverse=True)
    obs_parent_counts = np.bincount(inv, weights=counts).astype(np.int64)

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
    fitting strategy**, shared by :class:`AnDE` and :class:`WeightedAnDE`.

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
        Order ``n`` of the dependence, the number of super-parent attributes of
        each SPnDE. It must satisfy ``0 <= n_dependence <= n_features``.

        - ``n=0``: Naive Bayes.
        - ``n=1``: AODE (Averaged One-Dependence Estimators).
        - ``n=2``: A2DE.

    n_bins : int, default=5
        Number of bins for discretizing continuous attributes when they act as
        super-parents (at least 2). Children remain continuous and are modeled
        by Gaussian distributions.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for the discretization of super-parents.

    alpha : float, default=1.0
        Non-negative smoothing parameter passed to the internal
        :class:`MixedNB` estimators and used for the augmented-class prior.

    n_jobs : int, default=None
        Number of jobs used to fit the SPnDEs.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as categorical,
        passed to :class:`MixedNB`, which describes how the undeclared
        attributes are typed.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as Bernoulli.
        See :class:`MixedNB`.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as Gaussian.
        See :class:`MixedNB`.

    var_prior_weight : float, default=1.0
        Pseudo-observations shrinking each Gaussian child variance towards the
        pooled variance of the feature (see :class:`MixedNB`). It applies to
        the SPnDEs with super-parents and to the class-level back-off model;
        ``0`` gives the variances of :class:`~sklearn.naive_bayes.GaussianNB`.

    min_parent_count : int, default=1
        Minimum training frequency of a parent instantiation for its SPnDE to
        take part in the prediction of an instance. Values below 1 behave as 1.

    unseen_state : {"laplace", "backoff"}, default="laplace"
        Child conditionals used for an augmented state never seen in training.
        ``"laplace"`` is the standard smoothed estimate of an empty cell;
        ``"backoff"`` reuses the class-level conditionals. An ablation over the
        benchmark found no practical difference between them.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    class_count_ : ndarray of shape (n_classes,)
        Number of training samples of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ensemble_ : list of dict
        The fitted SPnDEs, one per set of ``n_dependence`` super-parents (a
        single one for ``n_dependence=0``). Each dict holds the
        ``"parent_indices"`` and ``"child_indices"`` of the component, its
        ``"estimator"`` (a :class:`MixedNB` fitted on the augmented class
        :math:`Y^*`) and the counts used for the minimum-frequency rule.

    backoff_ : MixedNB
        Naive Bayes on the original class, used for the class-level
        conditionals, to resolve the type of every attribute and as the
        fallback when no SPnDE qualifies.
    """

    _parameter_constraints: dict = {
        "n_dependence": [Interval(Integral, 0, None, closed="left")],
        "n_bins": [Interval(Integral, 2, None, closed="left")],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "n_jobs": [Integral, None],
        "categorical_features": ["array-like", None],
        "bernoulli_features": ["array-like", None],
        "gaussian_features": ["array-like", None],
        "var_prior_weight": [Interval(Real, 0, None, closed="left")],
        "min_parent_count": [Interval(Integral, 0, None, closed="left")],
        "unseen_state": [StrOptions({"laplace", "backoff"})],
    }

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

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the SPnDEs of the ensemble by counting frequencies.

        Learns the joint probability :math:`P(y, \\mathbf{x})` of every SPnDE
        and the class-level back-off model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        if (
            not isinstance(self.n_dependence, numbers.Integral)
            or not 0 <= self.n_dependence <= self.n_features_in_
        ):
            raise ValueError(
                "n_dependence must be an integer between 0 and the number of"
                f" features ({self.n_features_in_}), got {self.n_dependence!r}"
            )
        if self.unseen_state not in ("backoff", "laplace"):
            raise ValueError(
                "unseen_state must be 'backoff' or 'laplace', got"
                f" {self.unseen_state!r}"
            )

        # --- 1. Class-level back-off model (Naive Bayes on the original class),
        #        which also resolves the type of every attribute ---
        self.backoff_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=self.gaussian_features,
            var_prior_weight=self.var_prior_weight,
        ).fit(X, y)
        types = self.backoff_.feature_types_

        # --- 2. Discretize parents, with the types the children use ---
        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=types["gaussian"],
            discrete_features=types["categorical"] + types["bernoulli"],
        )
        parent_data = self._encoder.fit_transform(X)
        gauss = types["gaussian"]
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

        # Work shared by every component, computed once: class indices and
        # the categories and codes of the categorical attributes.
        y_int = np.searchsorted(self.classes_, y)
        self.class_count_ = np.bincount(y_int, minlength=len(self.classes_)).astype(
            float
        )
        X_int, categories = self.backoff_._shared_encoding(X)

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
                y_int,
                parent_data,
                self.n_features_in_,
                self.alpha,
                self._encoder.cards_,
                types["categorical"],
                types["bernoulli"],
                types["gaussian"],
                self.var_prior_weight,
                len(self.classes_),
                X_int,
                categories,
            )
            for p_idx in parent_combinations
        )

        return self

    def _parent_keys(self, X_parents_disc):
        """Linear index of the parent instantiation of every component,
        shape (n_samples, n_models); 0 for components without parents."""
        keys = np.zeros((X_parents_disc.shape[0], len(self.ensemble_)), dtype=np.int64)
        for m_idx, info in enumerate(self.ensemble_):
            if info["parent_indices"]:
                keys[:, m_idx] = (
                    X_parents_disc[:, info["parent_indices"]] @ info["parent_strides"]
                )
        return keys

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
        if types["categorical"]:
            # A value never seen in training contributes nothing
            unseen = self.backoff_._categorical_codes(X)[1]
            out[:, types["categorical"]] = np.where(
                unseen, 0.0, -np.log(self.backoff_._cat_cardinalities)
            )
        if types["bernoulli"]:
            out[:, types["bernoulli"]] = -np.log(2.0)
        return out

    def _get_jll_per_model(self, X, per_factor=False):
        """
        Computes the log-probability log P_m(y, x) for each model 'm' in the ensemble.

        Parameters
        ----------
        per_factor : bool, default=False
            If True, return the factors of each joint instead of their sum:
            the root ``log P(y, x_p)`` followed by ``log P(x_i | y, x_p)`` for
            every child ``i`` in ``child_indices`` order.

        Returns
        -------
        jll_tensor : ndarray of shape (n_samples, n_classes, n_models)
            Contains log P(y, x) according to each SPODE. With ``per_factor``
            it has an extra last axis of length ``1 + n_children``.

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
        return self._jll_per_model(validate_data(self, X, reset=False), per_factor)

    def _jll_per_model(self, X, per_factor=False):
        """:meth:`_get_jll_per_model` on an already validated ``X``."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_models = len(self.ensemble_)

        X_parents_disc, unseen = self._encoder.transform(X)
        parent_keys = self._parent_keys(X_parents_disc)

        # --- Pass 1: activity and augmented-state hits ---
        hits, positions = [], []
        active = np.ones((n_samples, n_models), dtype=bool)
        needs_backoff = np.zeros(n_samples, dtype=bool)
        all_classes = np.broadcast_to(np.arange(n_classes), (n_samples, n_classes))
        for m_idx, info in enumerate(self.ensemble_):
            if self.n_dependence == 0:
                # Plain Naive Bayes: its augmented states are the classes
                hits.append(np.ones((n_samples, n_classes), dtype=bool))
                positions.append(all_classes)
                continue
            pk = parent_keys[:, m_idx]
            pos_p, found_p = _lookup(info["obs_parent_keys"], pk)
            counts = np.where(found_p, info["obs_parent_counts"][pos_p], 0)
            active[:, m_idx] = (counts >= max(self.min_parent_count, 1)) & ~unseen[
                :, info["parent_indices"]
            ].any(axis=1)

            queries = all_classes * info["n_parent_combos"] + pk[:, None]
            pos, hit = _lookup(info["ystar_keys"], queries)
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
        n_factors = 1 + self.n_features_in_ - self.n_dependence
        out = np.zeros(
            (n_samples, n_classes, n_models) + ((n_factors,) if per_factor else ())
        )
        for m_idx, info in enumerate(self.ensemble_):
            child_indices = info["child_indices"]
            est = info["estimator"]
            X_sub = X[:, child_indices] if child_indices else np.zeros((n_samples, 1))
            hit, pos = hits[m_idx], positions[m_idx]

            if per_factor:
                factors = out[:, :, m_idx]
                factors[..., 0] = np.where(hit, est.class_log_prior_[pos], 0.0)
                if child_indices:
                    child = est._feature_log_likelihood(X_sub, class_idx=pos)
                    factors[..., 1:] = np.where(hit[..., None], child, 0.0)
            else:
                # Only the augmented states of the classes are evaluated, not
                # every state of the component
                jll = est.class_log_prior_[pos]
                if child_indices:
                    child = est._feature_log_likelihood(X_sub, class_idx=pos)
                    jll = jll + child.sum(axis=2)
                out[:, :, m_idx] = np.where(hit, jll, 0.0)

            miss = active[:, m_idx][:, None] & ~hit
            if miss.any():
                r_idx, c_idx = np.nonzero(miss)
                if self.unseen_state == "laplace":
                    uninf = self._uninformed_log_likelihood(X[r_idx])
                    child_ll = uninf[:, child_indices]
                else:
                    child_ll = feat_ll[row_map[r_idx], c_idx][:, child_indices]
                if per_factor:
                    out[r_idx, c_idx, m_idx, 0] = info["log_prior_unseen"]
                    out[r_idx, c_idx, m_idx, 1:] = child_ll
                else:
                    out[r_idx, c_idx, m_idx] = info["log_prior_unseen"] + child_ll.sum(
                        axis=1
                    )

        return out, X_parents_disc, active, nb_jll

    @staticmethod
    def _normalize(scores):
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        log_prob = np.nan_to_num(log_prob, nan=-np.log(scores.shape[1]))
        # Below the smallest positive float the probability would underflow to
        # 0, and predict_log_proba would no longer be log(predict_proba)
        return np.maximum(log_prob, _LOG_TINY)

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Probability of the samples for each class, in the order of
            ``classes_``.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# =============================================================================
# 1. Generative model
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


_POOLS = {"arithmetic": _arithmetic_scores, "geometric": _geometric_scores}


class AnDE(_BaseAnDE):
    """
    Averaged n-Dependence Estimators (AnDE) [Generative].

    This is the generative model described by Webb et al. [1]_: an ensemble of
    SPnDEs whose joint probabilities are pooled with the **arithmetic mean**,

    .. math::
        P(y|x) \\propto \\frac{1}{|M|} \\sum_{m} P_m(y, x).

    With ``mean="geometric"`` the joints are pooled with the geometric mean
    instead, :math:`P(y|x) \\propto \\prod_m P_m(y, x)^{1/|M|}`. This
    logarithmic pool of SPnDEs is not the AnJE of Zaidi et al. [3]_, which
    pools n-joins :math:`P(x_\\alpha | y)` (see :class:`skbn.AnJE`); for
    ``n_dependence=1`` it equals
    :math:`P(y) \\prod_{i<j} P(x_i, x_j | y)^{2/a} \\prod_i P(x_i | y)^{-(a-2)/a}`
    on discrete data.

    This implementation extends the original AnDE by supporting **mixed data types**
    (Gaussian/Categorical) through the Super-Class strategy.

    When a test instance falls in an augmented state :math:`(y, \\mathbf{x}_p)`
    never seen in training, the child conditionals come from ``unseen_state``;
    a SPnDE whose parent instantiation occurred fewer than ``min_parent_count``
    times is left out of the pool for that instance, and the prediction falls
    back to Naive Bayes when no SPnDE qualifies.

    Parameters
    ----------
    n_dependence : int, default=1
        Order ``n`` of the dependence, the number of super-parent attributes of
        each SPnDE. It must satisfy ``0 <= n_dependence <= n_features``.

        - ``n=0``: Naive Bayes.
        - ``n=1``: AODE (Averaged One-Dependence Estimators).
        - ``n=2``: A2DE.

    n_bins : int, default=5
        Number of bins for discretizing continuous attributes when they act as
        super-parents (at least 2). Children remain continuous and are modeled
        by Gaussian distributions.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for the discretization of super-parents.

    alpha : float, default=1.0
        Non-negative smoothing parameter passed to the internal
        :class:`MixedNB` estimators and used for the augmented-class prior.

    n_jobs : int, default=None
        Number of jobs used to fit the SPnDEs.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as categorical,
        passed to :class:`MixedNB`, which describes how the undeclared
        attributes are typed.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as Bernoulli.
        See :class:`MixedNB`.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as Gaussian.
        See :class:`MixedNB`.

    var_prior_weight : float, default=1.0
        Pseudo-observations shrinking each Gaussian child variance towards the
        pooled variance of the feature (see :class:`MixedNB`). It applies to
        the SPnDEs with super-parents and to the class-level back-off model;
        ``0`` gives the variances of :class:`~sklearn.naive_bayes.GaussianNB`.

    min_parent_count : int, default=1
        Minimum training frequency of a parent instantiation for its SPnDE to
        take part in the prediction of an instance. Values below 1 behave as 1.

    unseen_state : {"laplace", "backoff"}, default="laplace"
        Child conditionals used for an augmented state never seen in training.
        ``"laplace"`` is the standard smoothed estimate of an empty cell
        (uniform for discrete children, the pooled Gaussian of the attribute
        for continuous ones); ``"backoff"`` reuses the class-level
        conditionals. An ablation over the benchmark found no practical
        difference between them.

    mean : {"arithmetic", "geometric"}, default="arithmetic"
        Pooling of the components' joint probabilities.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    class_count_ : ndarray of shape (n_classes,)
        Number of training samples of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ensemble_ : list of dict
        The fitted SPnDEs, one per set of ``n_dependence`` super-parents (a
        single one for ``n_dependence=0``). Each dict holds the
        ``"parent_indices"`` and ``"child_indices"`` of the component, its
        ``"estimator"`` (a :class:`MixedNB` fitted on the augmented class
        :math:`Y^*`) and the counts used for the minimum-frequency rule.

    backoff_ : MixedNB
        Naive Bayes on the original class, used for the class-level
        conditionals, to resolve the type of every attribute and as the
        fallback when no SPnDE qualifies.

    See Also
    --------
    WeightedAnDE : AnDE with discriminatively learned weights.
    AnJE : Averaged n-Join Estimators.
    MixedNB : Naive Bayes classifier for mixed data.

    References
    ----------
    .. [1] Webb, G. I., Boughton, J., & Wang, Z. (2005). Not so naive Bayes:
           Aggregating one-dependence estimators. Machine Learning, 58(1), 5-24.
    .. [2] Webb, G. I., Boughton, J., Zheng, F., Ting, K. M., & Salem, H. (2011).
           Learning by extrapolation from marginal to full-multivariate
           probability distributions: Decreasingly naive Bayesian
           classification. Machine Learning, 86(2), 233-272.
    .. [3] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., &
           Cerquides, J. (2016). ALR^n: Accelerated higher-order logistic
           regression. Machine Learning, 104(2-3), 151-194.
    """

    _parameter_constraints: dict = {
        **_BaseAnDE._parameter_constraints,
        "mean": [StrOptions({"arithmetic", "geometric"})],
    }

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
        mean="arithmetic",
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
        self.mean = mean

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the ensemble of SPnDEs by counting frequencies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.mean not in _POOLS:
            raise ValueError(
                f"mean must be 'arithmetic' or 'geometric', got {self.mean!r}"
            )
        return super().fit(X, y)

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Log-probability of the samples for each class, in the order of
            ``classes_``. The estimates are normalised over the classes and
            bounded below by the logarithm of the smallest positive float.
        """
        jll, _, active, nb_jll = self._get_jll_per_model(X)
        return self._normalize(_POOLS[self.mean](jll, active, nb_jll))


# =============================================================================
# 2. Discriminative weighting
# =============================================================================

# Variables that index a weight at each level. A weight always belongs to one
# term (a component, or a factor of a component); "value" is the value of the
# term's own variable, "class" the class and "parents" the parent
# instantiation of a child factor.
_LEVEL_DIMS = {
    "component": {
        1: (),
        2: ("value",),
        3: ("class",),
        4: ("value", "class"),
    },
    "factor": {
        1: (),
        2: ("value",),
        3: ("value", "class"),
        4: ("value", "class", "parents"),
    },
}


class _Design(NamedTuple):
    """Linear map from the weights to the component scores.

    The score of component ``m`` for sample ``i`` and class ``c`` is
    ``u[i, c, m] = intercept[i, c, m] + (operator @ theta)[(i * C + c) * M + m]``.
    Each operator row holds the log-terms of a component at the indices of
    their weights (exponent weights), or a one (mixture weights, with the
    log-joint as intercept). ``active[i, m]`` marks the components that take
    part in the prediction of sample ``i``.
    """

    operator: csr_matrix
    intercept: Optional[np.ndarray]
    active: np.ndarray

    def component(self, m_idx, mask, lo, hi):
        """Component ``m_idx`` alone, on the samples in ``mask`` and with its
        weights ``theta[lo:hi]``."""
        n_samples, n_models = self.active.shape
        n_classes = self.operator.shape[0] // max(n_samples * n_models, 1)
        samples = np.flatnonzero(mask)
        rows = (samples[:, None] * n_classes + np.arange(n_classes)) * n_models + m_idx
        return _Design(
            self.operator[rows.ravel()][:, lo:hi],
            None if self.intercept is None else self.intercept[mask][:, :, [m_idx]],
            np.ones((len(samples), 1), dtype=bool),
        )


class WeightedAnDE(AnDE):
    """
    Weighted Averaged n-Dependence Estimators (WeightedAnDE) [Hybrid].

    Discriminative weights learned on top of the generative AnDE by maximising
    the conditional log-likelihood with L-BFGS-B, pre-conditioned by the
    generative parameters (the weights start at, and are regularized towards,
    the generative model). ``weight_scope`` sets what each weight multiplies
    and ``mean`` how the components are pooled.

    **Component weights** (``weight_scope="component"``) weight each SPnDE as a
    whole. With the arithmetic mean they are mixture coefficients,

    .. math::
        s(y, x) = \\log \\sum_m w_m(y, x_p)\\, P_m(y, x),

    optimized in log-space (``w = exp(theta)``); since
    :math:`P_m = P(y, x_p) \\prod_i P(x_i | y, x_p)`, level 4 is a
    discriminative re-estimation of each component's root table
    :math:`P(y, x_p)`, and the lower levels tie its entries. Level 3 is the
    per-SPODE, per-class weighting of EODE-CLL [2]_ (without its attribute
    weights). With the geometric mean they scale each component's log-joint,
    :math:`s = \\frac{1}{|M|} \\sum_m w_m \\log P_m(y, x)`, with non-negative
    weights. Component weights do not change how a component combines its
    attributes, so they cannot correct a dependence between children shared
    by every component (e.g. a duplicated attribute).

    **Factor weights** (``weight_scope="factor"``) apply the weighted CCBN of
    Zaidi et al. [1]_ inside each SPnDE: every factor ``f`` of component ``m``
    gets its own exponent, :math:`P_m^w(y, x) = \\prod_f \\theta_f^{w_f}`, and
    the components are pooled as above. The root factor
    :math:`P(y, x_p)` is weighted as one factor whose value is the parent
    instantiation.

    With ``modular=True`` the weights of each SPnDE are fitted on that
    component's own conditional log-likelihood, a convex problem. Factor
    weights at level 4, fitted modularly and pooled with
    ``average="posterior"``, are the AnDE^w of FedAnDE [3]_: one wCCBN per
    SPnDE, whose posteriors are averaged. The joint fit of an arithmetic
    mixture is non-convex; the geometric mean gives a convex joint problem.

    Weight levels (variables indexing each weight, from coarsest to finest):

    ===== ======================================= ===================================
    Level ``weight_scope="component"``            ``weight_scope="factor"`` [1]_
    ===== ======================================= ===================================
    1     Component                               Factor
    2     Component x parent value                Factor x value
    3     Component x class                       Factor x value x class
    4     Component x class x parent value        Factor x value x class x parents
    ===== ======================================= ===================================

    The value of a Gaussian child is not discretized, so its factor weights
    ignore the "value" dimension. Only the combinations that occur in training
    get a weight; the others share one weight that keeps its generative value,
    so the number of weights is bounded by the data, not by the cardinalities
    of the attributes. Factor weights keep the per-factor log-likelihoods of
    the training set in memory, about ``12 * n_samples * n_classes *
    n_models * (n_features - n_dependence + 1)`` bytes.

    Parameters
    ----------
    n_dependence : int, default=1
        Order ``n`` of the dependence, the number of super-parent attributes of
        each SPnDE. It must satisfy ``0 <= n_dependence <= n_features``; ``0``
        gives the Naive Bayes topology.

    n_bins : int, default=5
        Number of bins for discretizing continuous attributes when they act as
        super-parents (at least 2). Children remain continuous and are modeled
        by Gaussian distributions.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for the discretization of super-parents.

    alpha : float, default=1.0
        Non-negative additive (Laplace/Lidstone) smoothing parameter for the
        probabilities of the generative model.

    n_jobs : int, default=None
        Number of jobs used to fit the SPnDEs of the generative phase; the
        weights are optimized in a single process.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as categorical,
        passed to :class:`MixedNB`, which describes how the undeclared
        attributes are typed.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as Bernoulli.
        See :class:`MixedNB`.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as Gaussian.
        See :class:`MixedNB`.

    var_prior_weight : float, default=1.0
        Pseudo-observations shrinking each Gaussian child variance towards the
        pooled variance of the feature (see :class:`MixedNB`). It applies to
        the SPnDEs with super-parents and to the class-level back-off model;
        ``0`` gives the variances of :class:`~sklearn.naive_bayes.GaussianNB`.

    min_parent_count : int, default=1
        Minimum training frequency of a parent instantiation for its SPnDE to
        take part in the prediction of an instance. Values below 1 behave as 1.

    unseen_state : {"laplace", "backoff"}, default="laplace"
        Child conditionals used for an augmented state never seen in training.
        ``"laplace"`` is the standard smoothed estimate of an empty cell
        (uniform for discrete children, the pooled Gaussian of the attribute
        for continuous ones); ``"backoff"`` reuses the class-level
        conditionals.

    mean : {"arithmetic", "geometric"}, default="arithmetic"
        Pooling of the (weighted) components.

    average : {"joint", "posterior"}, default="joint"
        What the arithmetic mean pools: the components' weighted joints
        :math:`P^w_m(y, x)`, as in AnDE, or their posteriors
        :math:`P^w_m(y | x)`. Both give the same geometric mean. With
        ``"posterior"`` a class-independent component weight cancels, so
        arithmetic component weights need a class-specific level (3 or 4).

    weight_scope : {"component", "factor"}, default="component"
        Whether a weight multiplies a whole component or one of its factors.

    weight_level : int, default=1
        Granularity of the weights, from 1 (coarsest) to 4 (finest); see the
        table above.

    l2_reg : float, default=1.0
        Non-negative L2 regularization towards the generative solution
        (``w = 1``), added to the summed conditional log-likelihood.

    max_iter : int, default=100
        Maximum number of L-BFGS-B iterations (at least 1); with
        ``modular=True`` it applies to the fit of each component. A
        :class:`~sklearn.exceptions.ConvergenceWarning` is raised at the end of
        :meth:`fit` when L-BFGS-B stops at ``max_iter`` before converging;
        increase ``max_iter`` or ``tol`` in that case.

    tol : float or None, default=None
        Stopping tolerance of L-BFGS-B on the relative reduction of the
        objective and on the projected gradient; None uses SciPy's defaults.

    modular : bool, default=False
        If True, the weights of each component are fitted independently on
        that component's own conditional log-likelihood, which is convex. With
        arithmetic component weights only the class-specific levels (3 and 4)
        are identifiable; levels 1 and 2 reduce to the generative AnDE, the
        weights stay at 1 and a :class:`UserWarning` is raised.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    class_count_ : ndarray of shape (n_classes,)
        Number of training samples of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ensemble_ : list of dict
        The fitted SPnDEs of the generative phase, one per set of
        ``n_dependence`` super-parents (a single one for ``n_dependence=0``).
        Each dict holds the ``"parent_indices"`` and ``"child_indices"`` of the
        component, its ``"estimator"`` (a :class:`MixedNB` fitted on the
        augmented class :math:`Y^*`) and the counts used for the
        minimum-frequency rule.

    backoff_ : MixedNB
        Naive Bayes on the original class, used for the class-level
        conditionals, to resolve the type of every attribute and as the
        fallback when no SPnDE qualifies.

    coef_theta_ : ndarray of shape (n_weights,)
        Optimized parameters: ``log w`` for arithmetic component weights
        (``weight_scope="component"`` and ``mean="arithmetic"``), ``w``
        otherwise.

    learned_weights_ : ndarray of shape (n_weights,)
        Learned weights ``w``.

    n_weights_ : int
        Number of weights, including the shared one for unseen combinations.

    n_iter_ : int
        Total number of L-BFGS-B iterations (summed over the components when
        ``modular=True``).

    See Also
    --------
    AnDE : The generative model.
    GaussianALR : ALR^n on the factors of the SPnDEs.

    References
    ----------
    .. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
           Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
           Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.
    .. [2] Zheng, X., Lin, Z., Xu, H., Chen, C., & Ye, T. (2015). Efficient
           learning ensemble SuperParent-one-dependence estimator by maximizing
           conditional log likelihood. Expert Systems with Applications, 42,
           7732-7745.
    .. [3] Torrijos, P., Gámez, J. A., & Puerta, J. M. Federated Averaged
           n-Dependence Estimators. Manuscript, 2026.
    """

    _parameter_constraints: dict = {
        **AnDE._parameter_constraints,
        "average": [StrOptions({"joint", "posterior"})],
        "weight_scope": [StrOptions({"component", "factor"})],
        "weight_level": [Interval(Integral, 1, 4, closed="both")],
        "l2_reg": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "modular": ["boolean"],
    }

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
        mean="arithmetic",
        average="joint",
        weight_scope="component",
        weight_level=1,
        l2_reg=1.0,
        max_iter=100,
        tol=None,
        modular=False,
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
            mean,
        )
        self.average = average
        self.weight_scope = weight_scope
        self.weight_level = weight_level
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol
        self.modular = modular

    # --- Parameterization ---------------------------------------------------
    @property
    def _offset(self):
        """Whether weights are mixture coefficients (added in log-space)
        rather than exponents of the log-terms."""
        return self.weight_scope == "component" and self.mean == "arithmetic"

    @property
    def _pools_posteriors(self):
        """Whether the arithmetic mean pools the components' posteriors (the
        geometric mean of posteriors equals that of joints)."""
        return self.average == "posterior" and self.mean == "arithmetic"

    def _to_weights(self, theta):
        return np.exp(theta) if self._offset else theta

    def _children(self):
        """Child attributes of every component, shape (n_models, n_children)."""
        children = [info["child_indices"] for info in self.ensemble_]
        return np.array(children, dtype=np.int64).reshape(len(children), -1)

    def _child_value_codes(self, X):
        """Value code of every attribute as a child: the category of a
        categorical attribute, the bit of a Bernoulli one and 0 for Gaussian
        (or ignored) attributes, whose values are not discretized. A category
        never seen in training gets code 0; its factor is 0, so its weight
        does not matter."""
        types = self.backoff_.feature_types_
        codes = np.zeros(X.shape, dtype=np.int64)
        if types["categorical"]:
            codes[:, types["categorical"]] = self.backoff_._categorical_codes(X)[0]
        if types["bernoulli"]:
            codes[:, types["bernoulli"]] = (
                X[:, types["bernoulli"]] > self.backoff_.bernoulli_thresholds_
            )
        return codes

    def _term_cardinalities(self):
        """Number of value and parent codes of every term, each of shape
        (n_models, n_terms)."""
        n_combos = np.array([info["n_parent_combos"] for info in self.ensemble_])
        if self.weight_scope == "component":
            # One term per component, valued by its parent instantiation
            return n_combos[:, None], np.ones((len(n_combos), 1), dtype=np.int64)
        types = self.backoff_.feature_types_
        child_cards = np.ones(self.n_features_in_, dtype=np.int64)
        if types["categorical"]:
            child_cards[types["categorical"]] = self.backoff_._cat_cardinalities
        child_cards[types["bernoulli"]] = 2
        # Term 0 is the root P(y, x_p): its value is the parent instantiation
        # and it has no parents. Terms 1.. are the children P(x_i | y, x_p).
        value_cards = np.column_stack([n_combos, child_cards[self._children()]])
        parent_cards = np.repeat(n_combos[:, None], value_cards.shape[1], axis=1)
        parent_cards[:, 0] = 1
        return value_cards, parent_cards

    def _term_codes(self, X, X_parents_disc):
        """Value and parent codes of every term, each of shape
        (n_samples, n_models, n_terms)."""
        parent_keys = self._parent_keys(X_parents_disc)[:, :, None]
        if self.weight_scope == "component":
            return parent_keys, np.zeros_like(parent_keys)
        child_codes = self._child_value_codes(X)[:, self._children()]
        value_codes = np.concatenate([parent_keys, child_codes], axis=2)
        parent_codes = np.broadcast_to(parent_keys, value_codes.shape).copy()
        parent_codes[:, :, 0] = 0
        return value_codes, parent_codes

    def _setup_weights(self, X):
        """Lay out the weights on a validated training ``X``.

        Every term owns a block of keys, contiguous per component, indexed by
        the dimensions of the level. Only the keys that occur in training get
        a weight, so their number is bounded by the data rather than by the
        cardinalities; one extra weight, never trained, stands for every key
        unseen in training and keeps its generative value.
        """
        dims = _LEVEL_DIMS[self.weight_scope][self.weight_level]
        n_classes = len(self.classes_)
        value_cards, parent_cards = self._term_cardinalities()

        # Strides of the dimensions, innermost first: class, value, parents
        size = np.ones_like(value_cards)
        self._class_stride = size * ("class" in dims)
        size = size * (n_classes if "class" in dims else 1)
        self._value_stride = size * ("value" in dims)
        size = size * (value_cards if "value" in dims else 1)
        self._parent_stride = size * ("parents" in dims)
        size = size * (parent_cards if "parents" in dims else 1)
        self._term_offsets = np.cumsum(size.ravel()).reshape(size.shape) - size

        X_parents_disc = self._encoder.transform(X)[0]
        seen = []
        for rows in _row_chunks(len(X), size.size):
            base = self._weight_base(X[rows], X_parents_disc[rows])
            seen.append(
                np.unique([base + c * self._class_stride for c in range(n_classes)])
            )
        self._weight_keys = np.unique(np.concatenate(seen))
        self.n_weights_ = len(self._weight_keys) + 1
        self._component_bounds = np.append(
            np.searchsorted(self._weight_keys, self._term_offsets[:, 0]),
            len(self._weight_keys),
        )

    def _weight_base(self, X, X_parents_disc):
        """Key of the class-0 weight of every (sample, component, term)."""
        value_codes, parent_codes = self._term_codes(X, X_parents_disc)
        return (
            self._term_offsets
            + value_codes * self._value_stride
            + parent_codes * self._parent_stride
        )

    def _weight_index(self, keys):
        """Weight of each key; keys unseen in training share the last one."""
        n_seen = len(self._weight_keys)
        pos = np.minimum(np.searchsorted(self._weight_keys, keys), n_seen - 1)
        return np.where(self._weight_keys[pos] == keys, pos, n_seen)

    def _design(self, X):
        """Weighted-score design for a validated ``X``, and the NB fallback.

        The per-term log-likelihoods become the operator's values in place,
        and the weight indices are built in row chunks, so the operator costs
        one float64 and one int32 per (sample, class, component, term).
        """
        per_factor = self.weight_scope == "factor"
        ll, X_parents_disc, active, nb_jll = self._jll_per_model(X, per_factor)
        n_samples, n_classes, n_models = ll.shape[:3]
        n_terms = ll.shape[3] if per_factor else 1
        nnz = ll.size
        idx_dtype = np.int32 if max(nnz, self.n_weights_) < 2**31 else np.int64

        indices = np.empty((n_samples, n_classes, n_models, n_terms), dtype=idx_dtype)
        for rows in _row_chunks(n_samples, n_classes * n_models * n_terms):
            base = self._weight_base(X[rows], X_parents_disc[rows])
            for c in range(n_classes):
                indices[rows, c] = self._weight_index(base + c * self._class_stride)

        # Exponent weights multiply the log-terms; mixture weights add to the
        # log-joint, which becomes the intercept
        data = np.ones(nnz) if self._offset else ll.reshape(-1)
        operator = csr_matrix(
            (
                data,
                indices.reshape(-1),
                np.arange(0, nnz + 1, n_terms, dtype=idx_dtype),
            ),
            shape=(n_samples * n_classes * n_models, self.n_weights_),
        )
        intercept = ll if self._offset else None
        return _Design(operator, intercept, active), nb_jll

    # --- Objective ----------------------------------------------------------
    def _forward(self, theta, design):
        """Pooled scores, shape (n, C), and what :meth:`_backward` needs.

        Samples without an active component score 0 for every class; they do
        not depend on the weights and are predicted by the NB fallback.
        """
        n_samples, n_models = design.active.shape
        u = (design.operator @ theta).reshape(n_samples, -1, n_models)
        if design.intercept is not None:
            u += design.intercept
        n_active = design.active.sum(axis=1)
        mask = design.active[:, None, :]

        if self.mean == "geometric":
            share = mask / np.maximum(n_active, 1)[:, None, None]
            return (u * share).sum(axis=2), (share, None)

        q = None
        if self._pools_posteriors:
            log_q = u - logsumexp(u, axis=1, keepdims=True)
            u, q = log_q, np.exp(log_q)
        masked = np.where(mask, u, -np.inf)
        scores = logsumexp(masked, axis=2)
        scores[n_active == 0] = 0.0
        with np.errstate(invalid="ignore"):
            resp = np.nan_to_num(np.exp(masked - scores[:, :, None]))
        return scores, (resp, q)

    @staticmethod
    def _backward(dscores, cache):
        """Derivative of the loss with respect to every component score
        ``u[i, c, m]``, given its derivative ``dscores`` with respect to the
        pooled scores."""
        share_or_resp, q = cache
        du = dscores[:, :, None] * share_or_resp
        if q is not None:
            # Through the per-component normalization log_q = u - logsumexp_c u
            du -= q * du.sum(axis=1, keepdims=True)
        return du

    def _nll_and_grad(self, theta, design, Y):
        """Conditional negative log-likelihood and its gradient. Rows of ``Y``
        set to zero leave their samples out of the objective."""
        scores, cache = self._forward(theta, design)
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        nll = -np.sum(log_prob * Y)
        dscores = np.exp(log_prob) * Y.sum(axis=1, keepdims=True) - Y
        grad = design.operator.T @ self._backward(dscores, cache).ravel()
        return nll, grad

    def _optimize(self, theta, design, Y):
        bounds = (
            [(0.0, None)] * theta.size
            if self.weight_scope == "component" and self.mean == "geometric"
            else None
        )
        theta, n_iter, converged = _fit_weights(
            lambda t: self._nll_and_grad(t, design, Y),
            theta,
            self.l2_reg,
            self.max_iter,
            log_space=self._offset,
            bounds=bounds,
            tol=self.tol,
        )
        self.n_iter_ += n_iter
        self._n_problems += 1
        self._n_failed += not converged
        return theta

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the generative ensemble, then learn the discriminative weights.

        The SPnDEs are fitted by counting frequencies and the weights are
        optimized with L-BFGS-B on the conditional log-likelihood, starting at
        the generative model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.weight_scope not in _LEVEL_DIMS:
            raise ValueError(
                "weight_scope must be 'component' or 'factor', got"
                f" {self.weight_scope!r}"
            )
        if self.weight_level not in _LEVEL_DIMS[self.weight_scope]:
            raise ValueError(f"weight_level must be in 1..4, got {self.weight_level}")
        if self.average not in ("joint", "posterior"):
            raise ValueError(
                f"average must be 'joint' or 'posterior', got {self.average!r}"
            )

        # 1. Generative phase (pre-conditioning)
        super().fit(X, y)

        # 2. Weight layout and design on the training set
        X = validate_data(self, X, reset=False)
        self._setup_weights(X)
        design, _ = self._design(X)

        # The generative fit has already validated y (and warned on a column
        # vector). Samples with no active component are predicted by the NB
        # fallback, do not depend on the weights and are left out.
        y_idx = np.searchsorted(self.classes_, column_or_1d(y))
        Y = np.eye(len(self.classes_))[y_idx] * design.active.any(axis=1)[:, None]

        theta = np.zeros(self.n_weights_) if self._offset else np.ones(self.n_weights_)
        self.n_iter_ = 0
        self._n_problems = self._n_failed = 0

        # 3. Discriminative phase
        if not self.modular:
            theta = self._optimize(theta, design, Y)
        elif (
            self._offset and "class" not in _LEVEL_DIMS["component"][self.weight_level]
        ):
            warnings.warn(
                (
                    "Modular WeightedAnDE with arithmetic component weights at"
                    f" weight_level {self.weight_level} is equivalent to the"
                    " generative AnDE: a class-independent weight cancels in the"
                    " per-component softmax, so it is not identifiable from that"
                    " component's conditional likelihood. Weights are fixed to 1."
                ),
                UserWarning,
            )
        else:
            bounds = self._component_bounds
            for m_idx, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
                rows = design.active[:, m_idx]
                if rows.any():
                    theta[lo:hi] = self._optimize(
                        theta[lo:hi], design.component(m_idx, rows, lo, hi), Y[rows]
                    )

        _warn_not_converged(self._n_failed, self._n_problems, self.max_iter)
        self.coef_theta_ = theta
        self.learned_weights_ = self._to_weights(theta)
        return self

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Log-probability of the samples for each class, in the order of
            ``classes_``, from the weighted components. The estimates are
            normalised over the classes and bounded below by the logarithm of
            the smallest positive float.
        """
        check_is_fitted(self, attributes=["classes_", "ensemble_", "coef_theta_"])
        X = validate_data(self, X, reset=False)
        n_classes, n_models = len(self.classes_), len(self.ensemble_)
        n_terms = 1 + self.n_features_in_ - self.n_dependence
        out = np.empty((X.shape[0], n_classes))
        for rows in _row_chunks(X.shape[0], n_classes * n_models * n_terms):
            design, nb_jll = self._design(X[rows])
            scores, _ = self._forward(self.coef_theta_, design)
            fallback = ~design.active.any(axis=1)
            scores[fallback] = nb_jll[fallback]
            out[rows] = self._normalize(scores)
        return out
