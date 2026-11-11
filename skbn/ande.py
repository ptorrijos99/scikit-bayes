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

import numbers
import warnings
from collections.abc import Sequence
from itertools import combinations
from numbers import Integral, Real
from typing import NamedTuple, Optional

import numpy as np
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
    _fold_padding,
    _pad_last,
    _row_chunks,
    _warn_not_converged,
)
from ._kernels import (
    _MAX_LIGHT,
    DISCRETE,
    GAUSSIAN,
    LOG_ZERO,
    EvaluationInputs,
    SPnDETables,
    _threads,
    _weighted_arithmetic_gradient,
    _weighted_arithmetic_scores,
    _weighted_geometric_gradient,
    _weighted_geometric_scores,
    component_scores,
    count_pairs,
    count_tables,
    count_triples,
    dense_estimates,
    gaussian_log_norms,
    index_states,
    pooled_scores,
)
from .mixed_nb import MixedNB

_LOG_TINY = np.log(np.finfo(float).tiny)

# Fraction of the largest variance of the Gaussian children added to their
# variances, the default of GaussianNB and MixedNB
_VAR_SMOOTHING = 1e-9


def _lookup(sorted_keys, queries):
    """Positions of ``queries`` in ``sorted_keys`` and a mask of hits."""
    pos = np.searchsorted(sorted_keys, queries)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    return pos, sorted_keys[pos] == queries


def _parent_strides(parents, cards):
    """Stride of every parent in the linear index of a parent instantiation
    (the first parent is the most significant), shape (n_models, n)."""
    strides = np.ones(parents.shape, dtype=np.int64)
    for q in range(parents.shape[1] - 2, -1, -1):
        strides[:, q] = strides[:, q + 1] * cards[parents[:, q + 1]]
    return strides


def _parent_frequencies(keys, count, n_combos, state_off):
    """Distinct parent instantiations of every component and their training
    frequencies, sorted per component, with the offsets of each component."""
    n_models = len(n_combos)
    comp = np.repeat(np.arange(n_models), np.diff(state_off))
    pkeys = keys % n_combos[comp]
    order = np.lexsort((pkeys, comp))
    comp, pkeys = comp[order], pkeys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (comp[1:] != comp[:-1]) | (pkeys[1:] != pkeys[:-1])
    starts = np.flatnonzero(first)
    freq = np.add.reduceat(count[order], starts) if len(starts) else np.zeros(0)
    offsets = np.searchsorted(comp[starts], np.arange(n_models + 1))
    return offsets.astype(np.int64), pkeys[starts], freq


def _constant_parent_groups(
    parents, parent_cards, informative, b_disc, kind, voff, cards
):
    """Groups of the components that differ only in which constant attributes
    are among their parents: those with a single value as parents, none as
    children (``informative`` is False) and a back-off estimate of exactly 0.
    Such an attribute adds exactly 0 to every score, and is 0 in the parent
    key of every training row.

    Returns the group of every component, the first component of every group
    (groups in the order of their first component) and the mask of constant
    attributes.
    """
    n_models = len(parents)
    const = (parent_cards == 1) & ~informative
    for j in np.flatnonzero(const & (kind == DISCRETE)):
        const[j] = not b_disc[:, voff[j] : voff[j] + cards[j]].any()
    if parents.shape[1] == 0 or not const[parents].any():
        return np.arange(n_models), np.arange(n_models), const
    canon = np.where(const[parents], -1, parents)
    canon.sort(axis=1)
    _, first, group = np.unique(canon, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[group.reshape(-1)], first[order], const


def _shared_layout(
    g_parents,
    g_combos,
    const,
    parent_codes,
    codes,
    parent_cards,
    cards,
    multi,
    doff,
    n_classes,
    n_samples,
):
    """Inputs of :func:`count_pairs` (components with one parent) or
    :func:`count_triples` (two), or None when the children cannot be counted
    by pairs or triples of attributes: a discrete child whose parent codes
    differ from its child codes, or an index of the dense rows by state
    larger than 2**25 entries.

    Returns the mask of the components whose parents are not constant (their
    discrete children are counted by pairs or triples), the parent codes of
    the non-constant attributes transposed and their cardinalities, the pairs
    of them that start a triple, the component of every attribute (one
    parent, shape (n_attrs, 1)) or pair of them (two) and the column of
    every one of them as a child (-1 when it is not a discrete child).
    """
    for j in multi:
        if parent_cards[j] != cards[j] or (
            codes is not parent_codes
            and not np.array_equal(parent_codes[:, j], codes[:, j])
        ):
            return None
    by_shared = ~const[g_parents].any(axis=1)
    if not by_shared.any() or (n_classes * g_combos[by_shared]).sum() > 1 << 25:
        return None
    attrs = np.flatnonzero(~const)
    n_attrs = len(attrs)
    position = np.full(len(const), -1)
    position[attrs] = np.arange(n_attrs)
    own = np.flatnonzero(by_shared)
    if g_parents.shape[1] == 1:
        pair_group = np.full((n_attrs, 1), -1, dtype=np.int64)
        pair_group[position[g_parents[own, 0]], 0] = own
    else:
        pair_group = np.full((n_attrs, n_attrs), -1, dtype=np.int64)
        pair_group[position[g_parents[own, 0]], position[g_parents[own, 1]]] = own
    first, second = np.triu_indices(n_attrs, 1)
    keep = second < n_attrs - 1
    child_col = np.full(n_attrs, -1, dtype=np.int64)
    is_multi = np.zeros(len(const), dtype=bool)
    is_multi[multi] = True
    child_col[is_multi[attrs]] = doff[attrs[is_multi[attrs]]]
    return (
        by_shared,
        np.ascontiguousarray(parent_codes[:, attrs].T),
        parent_cards[attrs].astype(np.int64),
        first[keep].astype(np.int64),
        second[keep].astype(np.int64),
        pair_group,
        child_col,
    )


class _Ensemble(Sequence):
    """The ``ensemble_`` of a fitted AnDE: a read-only sequence with the dict
    of every component, built when it is accessed. A list of dicts would hold
    the child indices of every component, ``C(d, n) * (d - n)`` Python
    integers."""

    def __init__(
        self, parents, n_combos, parent_cards, strides, n_states, log_prior_unseen, d
    ):
        self._parents = parents
        self._n_combos = n_combos
        self._parent_cards = parent_cards
        self._strides = strides
        self._n_states = n_states
        self._log_prior_unseen = log_prior_unseen
        self._d = d

    def __len__(self):
        return len(self._parents)

    def __getitem__(self, m):
        if isinstance(m, slice):
            return [self[i] for i in range(*m.indices(len(self)))]
        m = range(len(self))[m]  # Bounds check and negative indices
        parents = self._parents[m]
        has_parents = len(parents) > 0
        return {
            "parent_indices": tuple(int(p) for p in parents),
            "child_indices": np.setdiff1d(np.arange(self._d), parents).tolist(),
            "n_parent_combos": int(self._n_combos[m]) if has_parents else 1,
            "parent_cards": self._parent_cards[m] if has_parents else np.array([1]),
            "parent_strides": self._strides[m] if has_parents else np.array([0]),
            "n_states": int(self._n_states[m]),
            "log_prior_unseen": float(self._log_prior_unseen[m]),
        }

    def children(self):
        """Child attributes of every component, shape (n_models, n_children)."""
        is_child = np.ones((len(self), self._d), dtype=bool)
        is_child[np.arange(len(self))[:, None], self._parents] = False
        return np.nonzero(is_child)[1].reshape(len(self), -1)


class _ChildEncodingMixin:
    """Encoding of the attributes as children, shared by the models that keep
    their conditionals in the tables of :mod:`skbn._kernels` (the AnDE family,
    TAN and KDB). Needs the fitted ``backoff_``, ``_encoder``, ``_pooled_mean``
    and ``_pooled_var``."""

    def _layout(self):
        """Type, offset in the rows of the discrete tables and Gaussian column
        of every attribute, and the number of values of every discrete one."""
        types = self.backoff_.feature_types_
        d = self.n_features_in_
        kind = np.zeros(d, dtype=np.int64)
        cards = np.zeros(d, dtype=np.int64)
        cards[types["categorical"]] = getattr(self.backoff_, "_cat_cardinalities", [])
        cards[types["bernoulli"]] = 2
        discrete = np.flatnonzero(cards)
        kind[discrete] = DISCRETE
        kind[types["gaussian"]] = GAUSSIAN
        voff = np.zeros(d, dtype=np.int64)
        voff[discrete] = np.cumsum(cards[discrete]) - cards[discrete]
        gidx = np.zeros(d, dtype=np.int64)
        gidx[types["gaussian"]] = np.arange(len(types["gaussian"]))
        return kind, voff, gidx, cards

    def _value_codes(self, X, cat_codes=None, dtype=np.int64):
        """Value of every discrete attribute as a child (the category of a
        categorical attribute, the bit of a Bernoulli one), -1 for categories
        never seen in training and 0 for the other attributes. ``cat_codes``
        optionally holds the codes of the categorical attributes of a
        training ``X``, as the back-off model returns them."""
        types = self.backoff_.feature_types_
        codes = np.zeros(X.shape, dtype=dtype)
        if cat_codes is not None:
            codes[:, types["categorical"]] = cat_codes
        elif types["categorical"]:
            cat, unseen = self.backoff_._categorical_codes(X)
            codes[:, types["categorical"]] = np.where(unseen, -1, cat)
        if types["bernoulli"]:
            codes[:, types["bernoulli"]] = (
                X[:, types["bernoulli"]] > self.backoff_.bernoulli_thresholds_
            )
        return codes

    def _backoff_tables(self, kind, voff, cards, n_values):
        """The class-level Naive Bayes in the layout of the component tables:
        its log prior, discrete log-conditionals and Gaussian parameters
        (mean, variance and ``log(2 pi variance)``)."""
        types = self.backoff_.feature_types_
        est = self.backoff_.estimators_
        n_classes = len(self.classes_)
        disc = np.zeros((n_classes, n_values))
        for i, j in enumerate(types["categorical"]):
            disc[:, voff[j] : voff[j] + cards[j]] = est[
                "categorical"
            ].feature_log_prob_[i]
        for i, j in enumerate(types["bernoulli"]):
            log_p = est["bernoulli"].feature_log_prob_[:, i]
            disc[:, voff[j]] = np.log1p(-np.exp(log_p))
            disc[:, voff[j] + 1] = log_p
        disc = np.nan_to_num(disc, nan=LOG_ZERO, neginf=LOG_ZERO)
        if types["gaussian"]:
            mean, var = est["gaussian"].theta_, est["gaussian"].var_
        else:
            mean = var = np.zeros((n_classes, 0))
        logn = gaussian_log_norms(var)
        return self.backoff_.class_log_prior_, disc, mean, var, logn

    def _evaluation_inputs(self, X):
        """Per-sample inputs of the evaluation kernels for a validated ``X``:
        the child value codes, the Gaussian attributes, the uninformed
        log-likelihood of every attribute (the Laplace estimate of an empty
        cell: uniform for a discrete child, the pooled Gaussian of the
        attribute for a continuous one, 0 for an unseen category) and the
        parent codes with their unseen mask."""
        types = self.backoff_.feature_types_
        codes = self._value_codes(X)
        xg = np.ascontiguousarray(X[:, types["gaussian"]], dtype=np.float64)
        uninf = np.zeros(X.shape)
        if types["gaussian"]:
            diff = xg - self._pooled_mean
            uninf[:, types["gaussian"]] = -0.5 * (
                diff**2 / self._pooled_var + np.log(2.0 * np.pi * self._pooled_var)
            )
        kind, _, _, cards = self._layout()
        discrete = np.flatnonzero(kind == DISCRETE)
        uninf[:, discrete] = np.where(
            codes[:, discrete] < 0, 0.0, -np.log(cards[discrete].astype(float))
        )
        parent_codes, unseen = self._encoder.transform(X)
        return EvaluationInputs(codes, xg, uninf, parent_codes, unseen)


class _BaseAnDE(_ChildEncodingMixin, ClassifierMixin, BaseEstimator):
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

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Strategy used for the discretization of super-parents.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Non-negative Laplace smoothing of the estimates of the SPnDEs and of
        the augmented-class prior.

    n_jobs : int, default=None
        Number of threads used to count and evaluate the SPnDEs.
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
        take part in the prediction of an instance. ``0`` behaves as 1.

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

    ensemble_ : sequence of dict
        The fitted SPnDEs, one per set of ``n_dependence`` super-parents
        (a single one for ``n_dependence=0``). Each dict holds the
        ``"parent_indices"`` and ``"child_indices"`` of the component and the
        numbers of its parent instantiations (``"n_parent_combos"``) and of
        the augmented states :math:`Y^*` seen in training (``"n_states"``);
        the estimates of all the components are stored together in flat
        tables.

    backoff_ : MixedNB
        Naive Bayes on the original class, used for the class-level
        conditionals, to resolve the type of every attribute and as the
        fallback when no SPnDE qualifies.
    """

    _parameter_constraints: dict = {
        "n_dependence": [Interval(Integral, 0, None, closed="left")],
        "n_bins": [Interval(Integral, 2, None, closed="left")],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans", "mdlp"})],
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

        # --- 1. Class-level back-off model (Naive Bayes on the original class),
        #        which also resolves the type of every attribute ---
        #        (X and y are validated; the codes of the categorical attributes
        #        serve the parents and the children as well)
        y_int = np.searchsorted(self.classes_, y)
        self.backoff_, cat_codes = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=self.gaussian_features,
            var_prior_weight=self.var_prior_weight,
        )._fit_encoded(X, y_int, self.classes_, return_codes=True)
        types = self.backoff_.feature_types_

        # --- 2. Discretize parents, with the types the children use ---
        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=types["gaussian"],
            discrete_features=types["categorical"] + types["bernoulli"],
        )
        # The codes of the training samples, in 16 bits for nearly every
        # dataset (a single type, so that the kernels compile once)
        parent_codes = self._encoder.fit_transform(
            X,
            y_int,
            known=(types["categorical"], self.backoff_.categories_, cat_codes),
            dtype=None,
        )
        code_type = parent_codes.dtype
        gauss = types["gaussian"]
        Xg = np.asarray(X[:, gauss], dtype=float) if gauss else np.zeros((len(X), 0))
        self._pooled_mean = Xg.mean(axis=0) if gauss else np.zeros(0)
        self._pooled_var = Xg.var(axis=0) + (
            self.backoff_.estimators_["gaussian"].epsilon_ if gauss else 0.0
        )

        # --- 3. Count the SPnDEs ---
        self.class_count_ = np.bincount(y_int, minlength=len(self.classes_)).astype(
            float
        )
        if len(types["categorical"]) == X.shape[1]:
            # Every attribute is categorical: its codes as a child are its
            # codes as a parent
            codes = parent_codes
        else:
            codes = self._value_codes(X, cat_codes, code_type)
        del cat_codes
        self._fit_components(X, y_int, parent_codes, codes)
        return self

    def _fit_components(self, X, y_int, parent_codes, codes):
        """Count every SPnDE and store its log-probabilities in flat tables
        (see :mod:`skbn._kernels`), one entry per augmented state seen in
        training, from the parent codes and the child value codes of the
        training samples."""
        types = self.backoff_.feature_types_
        n_samples, d = X.shape
        n_classes, n_dep = len(self.classes_), self.n_dependence
        alpha = self.alpha
        kind, voff, gidx, cards = self._layout()

        if n_dep:
            parents = np.array(list(combinations(range(d), n_dep)), dtype=np.int64)
        else:
            parents = np.zeros((1, 0), dtype=np.int64)
        parent_cards = self._encoder.cards_[parents]
        if n_classes * np.prod(parent_cards.astype(float), axis=1).max() >= 2**62:
            raise ValueError(
                "The augmented states of the SPnDEs exceed a 64-bit index; reduce"
                " n_bins or n_dependence."
            )
        n_combos = np.prod(parent_cards, axis=1).astype(np.int64)
        strides = _parent_strides(parents, self._encoder.cards_)

        # The children that carry information: a discrete attribute with a
        # single value has log P(x_j | s) = 0 in every state
        informative = (kind == GAUSSIAN) | ((kind == DISCRETE) & (cards > 1))
        attrs = np.flatnonzero(informative).astype(np.int64)
        multi = np.flatnonzero((kind == DISCRETE) & (cards > 1))
        dcol = np.full(d, -1, dtype=np.int64)
        dcol[multi] = np.arange(len(multi))
        doff = np.zeros(d, dtype=np.int64)
        doff[multi] = np.cumsum(cards[multi]) - cards[multi]
        n_columns = int(cards[multi].sum())
        group_cards, grp = np.unique(cards, return_inverse=True)
        b_root, b_disc, b_mean, b_var, b_logn = self._backoff_tables(
            kind, voff, cards, int(cards.sum())
        )

        # Components that differ only in which attributes with a single value
        # (as parents and as children) are among their parents have the same
        # tables and scores: each group of them is counted once, for its
        # first component
        group, rep, const = _constant_parent_groups(
            parents, self._encoder.cards_, informative, b_disc, kind, voff, cards
        )
        n_groups = len(rep)
        g_parents, g_strides, g_combos = parents[rep], strides[rep], n_combos[rep]

        # With one or two parents, the discrete children of the components
        # whose parents are not constant are counted by pairs or triples of
        # attributes
        shared = None
        if n_dep in (1, 2):
            shared = _shared_layout(
                g_parents,
                g_combos,
                const,
                parent_codes,
                codes,
                self._encoder.cards_,
                cards,
                multi,
                doff,
                n_classes,
                n_samples,
            )
        by_shared = np.zeros(n_groups, dtype=bool)
        # A pair or triple is counted in its own table when it has at most
        # half as many cells as samples, a pass over them that counts two or
        # three children
        max_cells = max(n_samples // 2, 1)
        if shared is not None:
            by_shared = shared[0]

        xg = np.ascontiguousarray(X[:, types["gaussian"]], dtype=np.float64)
        y_int = y_int.astype(np.int64)
        # The parent codes by attribute, read a column at a time
        parent_codes_t = np.ascontiguousarray(parent_codes.T)
        with _threads(self.n_jobs):
            sizes = index_states(
                y_int,
                parent_codes_t,
                g_parents,
                g_strides,
                g_combos,
                n_classes,
                _MAX_LIGHT,
            )
            state_off, dense_off, light_off = (
                np.concatenate([[0], np.cumsum(sizes[:, k])]).astype(np.int64)
                for k in range(3)
            )
            keys, count, drow, loff, dense, rows, mean, var = count_tables(
                y_int,
                parent_codes_t,
                codes,
                xg,
                g_parents,
                g_strides,
                g_combos,
                n_classes,
                _MAX_LIGHT,
                state_off,
                dense_off,
                light_off,
                attrs,
                kind,
                doff,
                gidx,
                n_columns,
                by_shared,
                self._encoder.cards_,
                max_cells,
            )
            if shared is not None:
                _, codes_t, ucards, first, second, pair_group, child_col = shared
                grid = n_classes * g_combos * by_shared
                moff = np.concatenate([[0], np.cumsum(grid)]).astype(np.int64)
                smap = np.full(moff[-1], -1, dtype=np.int32)
                state_group = np.repeat(np.arange(n_groups), sizes[:, 0])
                own = by_shared[state_group]
                smap[moff[state_group[own]] + keys[own]] = drow[own]
                if n_dep == 1:
                    count_pairs(
                        codes_t,
                        y_int,
                        n_classes,
                        ucards,
                        np.ascontiguousarray(pair_group[:, 0]),
                        child_col,
                        smap,
                        moff,
                        max_cells,
                        dense,
                    )
                else:
                    count_triples(
                        codes_t,
                        y_int,
                        n_classes,
                        ucards,
                        first,
                        second,
                        pair_group,
                        child_col,
                        smap,
                        moff,
                        max_cells,
                        dense,
                    )
                del smap
        n_states = sizes[:, 0]
        comp = np.repeat(np.arange(n_groups), n_states)

        # Root P(y, x_p), Laplace-smoothed over the full grid of augmented
        # states; without parents the component is Naive Bayes (MLE prior)
        if n_dep:
            denom = n_samples + alpha * n_classes * g_combos
            root = np.log((count + alpha) / denom[comp])
            with np.errstate(divide="ignore"):
                log_prior_unseen = np.log(alpha / denom)
        else:
            root = np.log(count / count.sum())
            log_prior_unseen = np.full(n_groups, -np.inf)

        # Discrete children: log((N(s, v) + alpha) / (N(s) + alpha * card)), in
        # the rows of the dense states and, for the light ones, in a table
        # indexed by N(s), the cardinality and N(s, v)
        # (both from the logarithms of every possible count and state total)
        dense_count = count[drow >= 0]
        with np.errstate(divide="ignore"):
            log_hits = np.log(np.arange(n_samples + 1, dtype=float) + alpha)
            log_total = np.log(dense_count[:, None] + alpha * group_cards[None, :])
            light = np.arange(_MAX_LIGHT + 1, dtype=float)
            lut = np.log(light[None, None, :] + alpha) - np.log(
                light[:, None, None] + alpha * group_cards[None, :, None]
            )
        with _threads(self.n_jobs):
            dense_estimates(
                dense, log_hits, log_total, np.repeat(grp[multi], cards[multi])
            )
        np.nan_to_num(lut, copy=False, nan=LOG_ZERO, neginf=LOG_ZERO)
        # Codes of the training rows of the light states, read when one is hit
        used, rows = np.unique(rows, return_inverse=True)
        rows = rows.astype(np.int32)
        tcodes = codes[np.ix_(used, multi)]

        # Gaussian children: the variances of GaussianNB (plus the largest
        # child variance times var_smoothing), shrunk towards the pooled
        # variance when the component has parents
        if xg.shape[1]:
            pooled = np.var(xg, axis=0)
            is_parent = np.zeros((n_groups, d), dtype=bool)
            is_parent[np.arange(n_groups)[:, None], g_parents] = True
            child_var = np.where(is_parent[:, types["gaussian"]], -np.inf, pooled)
            epsilon = _VAR_SMOOTHING * np.maximum(child_var.max(axis=1), 0.0)[comp]
            var += epsilon[:, None]
            nu = self.var_prior_weight
            if n_dep and nu > 0:
                n_c = count[:, None]
                var = (n_c * var + nu * (pooled + epsilon[:, None])) / (n_c + nu)

        # The attributes whose estimates for an unseen state can differ from
        # 0: the others (ignored, or discrete with a single value and a
        # back-off estimate of 0) are left out of its sum
        silent = (kind != DISCRETE) & (kind != GAUSSIAN)
        for j in np.flatnonzero((kind == DISCRETE) & (cards <= 1)):
            silent[j] = not b_disc[:, voff[j] : voff[j] + cards[j]].any()
        nonzero = np.flatnonzero(~silent).astype(np.int64)

        pk_off, pk_keys, pk_count = _parent_frequencies(
            keys, count, g_combos, state_off
        )
        self._tables = SPnDETables(
            parents=parents,
            strides=strides,
            n_combos=n_combos,
            group=group,
            rep=rep,
            const=const,
            state_off=state_off,
            state_keys=keys,
            root=root,
            count=count,
            drow=drow,
            loff=loff,
            dense=dense,
            rows=rows,
            tcodes=tcodes,
            lut=lut,
            mean=mean,
            var=var,
            logn=gaussian_log_norms(var),
            pk_off=pk_off,
            pk_keys=pk_keys,
            pk_count=pk_count,
            log_prior_unseen=log_prior_unseen,
            b_root=b_root,
            b_disc=b_disc,
            b_mean=b_mean,
            b_var=b_var,
            b_logn=b_logn,
            kind=kind,
            voff=voff,
            gidx=gidx,
            attrs=attrs,
            dcol=dcol,
            doff=doff,
            grp=grp.astype(np.int64),
            nonzero=nonzero,
        )
        self.ensemble_ = _Ensemble(
            parents,
            n_combos,
            parent_cards,
            strides,
            n_states[group],
            log_prior_unseen[group],
            d,
        )

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
        inputs = self._evaluation_inputs(X)
        with _threads(self.n_jobs):
            out, active, nb_jll = component_scores(
                inputs,
                self._tables,
                max(self.min_parent_count, 1),
                self.unseen_state == "backoff",
                per_factor,
            )
        return (out if per_factor else out[..., 0]), inputs.pcodes, active, nb_jll

    def _pooled_scores(self, X, geometric):
        """Arithmetic or geometric pool of the components' joints for a
        validated ``X``, with the Naive Bayes fallback, shape
        (n_samples, n_classes)."""
        with _threads(self.n_jobs):
            return pooled_scores(
                self._evaluation_inputs(X),
                self._tables,
                max(self.min_parent_count, 1),
                self.unseen_state == "backoff",
                geometric,
            )

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

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Strategy used for the discretization of super-parents.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Non-negative Laplace smoothing of the estimates of the SPnDEs and of
        the augmented-class prior.

    n_jobs : int, default=None
        Number of threads used to count and evaluate the SPnDEs.
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
        take part in the prediction of an instance. ``0`` behaves as 1.

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

    ensemble_ : sequence of dict
        The fitted SPnDEs, one per set of ``n_dependence`` super-parents
        (a single one for ``n_dependence=0``). Each dict holds the
        ``"parent_indices"`` and ``"child_indices"`` of the component and the
        numbers of its parent instantiations (``"n_parent_combos"``) and of
        the augmented states :math:`Y^*` seen in training (``"n_states"``);
        the estimates of all the components are stored together in flat
        tables.

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

    Examples
    --------
    >>> import numpy as np
    >>> from skbn import AnDE, MixedNB
    >>> rng = np.random.RandomState(0)
    >>> x0 = rng.randint(0, 2, size=200)
    >>> x1 = rng.randint(0, 2, size=200)
    >>> x2 = rng.randint(0, 3, size=200)  # noise
    >>> y = x0 ^ x1  # exclusive or: no attribute is informative on its own
    >>> X = np.column_stack([x0, x1, x2])
    >>> MixedNB().fit(X, y).score(X, y) < 0.6
    True
    >>> clf = AnDE(n_dependence=1).fit(X, y)
    >>> clf.score(X, y)
    1.0
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
        check_is_fitted(self, attributes=["classes_", "ensemble_"])
        X = validate_data(self, X, reset=False)
        return self._normalize(self._pooled_scores(X, self.mean == "geometric"))


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
    ``u[i, c, m] = intercept[i, c, m] + sum_t values[i, c, m, t] *
    theta[index[i, m, t] + c * class_step]``. Each term is a log-term of the
    component times its weight (exponent weights), or, without ``values``,
    the weight alone (mixture weights, with the log-joint as intercept).

    The weights of the classes of a term are consecutive (``class_step`` 1)
    or one and the same (0), so ``index`` holds the weight of class 0 only:
    an index per class would repeat it ``n_classes`` times.
    """

    index: np.ndarray  # (n_samples, n_models, n_terms)
    class_step: int
    values: Optional[np.ndarray]  # (n_samples, n_classes, n_models, n_terms)
    intercept: Optional[np.ndarray]  # (n_samples, n_classes, n_models)
    active: np.ndarray  # (n_samples, n_models): takes part in the prediction

    def component(self, m_idx, mask, lo):
        """Component ``m_idx`` alone, on the samples in ``mask`` and with its
        weights, which start at ``lo``."""

        def own(a):  # Its slice of an array with the models on the third axis
            return None if a is None else a[:, :, m_idx][mask][:, :, None]

        return _Design(
            self.index[:, m_idx][mask][:, None] - lo,
            self.class_step,
            own(self.values),
            own(self.intercept),
            np.ones((int(mask.sum()), 1), dtype=bool),
        )

    def kernel_args(self):
        """Arguments of the compiled scores and gradients: the index, the
        class step, the values and the number of terms."""
        values = np.zeros(0) if self.values is None else self.values
        return (
            np.ascontiguousarray(self.index).reshape(-1),
            self.class_step,
            np.ascontiguousarray(values).reshape(-1),
            self.index.shape[2],
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

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Strategy used for the discretization of super-parents.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Non-negative additive (Laplace/Lidstone) smoothing parameter for the
        probabilities of the generative model.

    n_jobs : int, default=None
        Number of threads used to count and evaluate the SPnDEs; the
        weights are optimized in a single thread.
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
        take part in the prediction of an instance. ``0`` behaves as 1.

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

    ensemble_ : sequence of dict
        The fitted SPnDEs of the generative phase, one per set of
        ``n_dependence`` super-parents (a single one for ``n_dependence=0``).
        Each dict holds the ``"parent_indices"`` and ``"child_indices"`` of the
        component and the numbers of its parent instantiations
        (``"n_parent_combos"``) and of the augmented states :math:`Y^*` seen in
        training (``"n_states"``); the estimates of all the components are
        stored together in flat tables.

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

    Examples
    --------
    >>> import numpy as np
    >>> from skbn import WeightedAnDE
    >>> rng = np.random.RandomState(0)
    >>> x0 = rng.randint(0, 2, size=200)
    >>> x1 = rng.randint(0, 2, size=200)
    >>> x2 = rng.randint(0, 3, size=200)  # noise
    >>> y = x0 ^ x1  # exclusive or: no attribute is informative on its own
    >>> X = np.column_stack([x0, x1, x2])
    >>> clf = WeightedAnDE().fit(X, y)
    >>> clf.learned_weights_.shape  # a weight per SPnDE and a shared one
    (4,)
    >>> clf.score(X, y)
    1.0
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

    # Projected-gradient tolerance (None: that of ``tol``) and number of
    # corrections of L-BFGS-B
    _gtol = None
    _maxcor = 10

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
        return self.ensemble_.children()

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
        the dimensions of the level, the class innermost. Only the keys that
        occur in training get a weight, so their number is bounded by the
        data rather than by the cardinalities; one extra weight, never
        trained, stands for every key unseen in training and keeps its
        generative value.

        A key that occurs is given a weight for every class, so the keys are
        collected for class 0 alone (``_base_keys``) and the weight of class
        ``c`` follows that of class 0 by ``c`` places.
        """
        dims = _LEVEL_DIMS[self.weight_scope][self.weight_level]
        n_classes = len(self.classes_)
        value_cards, parent_cards = self._term_cardinalities()
        self._class_step = int("class" in dims)
        per_key = n_classes if self._class_step else 1

        # Strides of the dimensions, innermost first: class, value, parents
        size = np.full_like(value_cards, per_key)
        self._value_stride = size * ("value" in dims)
        size = size * (value_cards if "value" in dims else 1)
        self._parent_stride = size * ("parents" in dims)
        size = size * (parent_cards if "parents" in dims else 1)
        self._term_offsets = np.cumsum(size.ravel()).reshape(size.shape) - size

        X_parents_disc = self._encoder.transform(X)[0]
        seen = [
            np.unique(self._weight_base(X[rows], X_parents_disc[rows]))
            for rows in _row_chunks(len(X), size.size)
        ]
        self._base_keys = np.unique(np.concatenate(seen))
        self._n_seen = len(self._base_keys) * per_key
        self.n_weights_ = self._n_seen + 1
        self._component_bounds = np.append(
            np.searchsorted(self._base_keys, self._term_offsets[:, 0]) * per_key,
            self._n_seen,
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
        """Class-0 weight of each class-0 key; the keys unseen in training
        share the last weight, whatever the class."""
        last = len(self._base_keys) - 1
        pos = np.minimum(np.searchsorted(self._base_keys, keys), last)
        per_key = len(self.classes_) if self._class_step else 1
        return np.where(self._base_keys[pos] == keys, pos * per_key, self._n_seen)

    def _design(self, X):
        """Weighted-score design for a validated ``X``, and the NB fallback.

        The per-term log-likelihoods become the values of the design in
        place, and the weight indices are built in row chunks, so the design
        costs one float64 per (sample, class, component, term) and one int32
        per (sample, component, term).
        """
        per_factor = self.weight_scope == "factor"
        ll, X_parents_disc, active, nb_jll = self._jll_per_model(X, per_factor)
        n_samples, _, n_models = ll.shape[:3]
        n_terms = ll.shape[3] if per_factor else 1
        idx_dtype = (
            np.int32 if self.n_weights_ + len(self.classes_) < 2**31 else np.int64
        )

        index = np.empty((n_samples, n_models, n_terms), dtype=idx_dtype)
        for rows in _row_chunks(n_samples, n_models * n_terms):
            index[rows] = self._weight_index(
                self._weight_base(X[rows], X_parents_disc[rows])
            )

        # Exponent weights multiply the log-terms; mixture weights add to the
        # log-joint, which becomes the intercept
        if self._offset:
            design = _Design(index, self._class_step, None, ll, active)
        else:
            values = ll.reshape(n_samples, -1, n_models, n_terms)
            design = _Design(index, self._class_step, values, None, active)
        return design, nb_jll

    # --- Objective ----------------------------------------------------------
    def _scores(self, theta, design):
        """Pooled scores of the samples of ``design``, shape (n, C), and, for
        the arithmetic pool, the component scores the gradient reuses.

        The geometric pool is ``sum_m share[i, m] * u[i, c, m]`` over the
        active components, the arithmetic one ``log sum_m exp(u[i, c, m])``
        (of the components' posteriors with ``average="posterior"``), with
        ``u`` as in :class:`_Design`. Samples without an active component
        score 0 for every class; they do not depend on the weights and are
        predicted by the NB fallback. Compiled kernels compute them sample by
        sample, without arrays of every sample, class and component but ``u``
        for the arithmetic pool, with the operations of a sparse product,
        NumPy's sums and SciPy's ``logsumexp`` (from SciPy 1.15) in the same
        order.
        """
        n_classes = len(self.classes_)
        intercept = design.intercept
        if intercept is None:
            intercept = np.zeros((0, 0, 0))
        args = (
            *design.kernel_args(),
            _pad_last(theta, design.class_step * (n_classes - 1)),
            intercept,
            design.active,
            n_classes,
        )
        if self.mean == "geometric":
            return _weighted_geometric_scores(*args), None
        return _weighted_arithmetic_scores(*args, self._pools_posteriors)

    def _nll_and_grad(self, theta, design, Y):
        """Conditional negative log-likelihood and its gradient. Rows of ``Y``
        set to zero leave their samples out of the objective."""
        scores, terms = self._scores(theta, design)
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        nll = -np.sum(log_prob * Y)
        dscores = np.exp(log_prob) * Y.sum(axis=1, keepdims=True) - Y
        n_classes = len(self.classes_)
        n_padded = theta.size + design.class_step * (n_classes - 1)
        args = design.kernel_args()
        if self.mean == "geometric":
            grad = _weighted_geometric_gradient(*args, dscores, design.active, n_padded)
        else:
            grad = _weighted_arithmetic_gradient(
                *args,
                terms,
                design.active,
                scores,
                dscores,
                self._pools_posteriors,
                n_padded,
            )
        return nll, _fold_padding(grad, theta.size)

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
            gtol=self._gtol,
            maxcor=self._maxcor,
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
                        theta[lo:hi], design.component(m_idx, rows, lo), Y[rows]
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
            scores = self._scores(self.coef_theta_, design)[0]
            fallback = ~design.active.any(axis=1)
            scores[fallback] = nb_jll[fallback]
            out[rows] = self._normalize(scores)
        return out
