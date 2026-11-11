"""
Ensembles of SuperParent n-Dependence Estimators (SPnDEs): the base of the
AnDE family.

``_BaseAnDE`` fits the generative tables of every SPnDE of the ensemble,
with the "Super-Class" strategy for mixed data, and evaluates them. It is
the base of :class:`~skbn.AnDE` and :class:`~skbn.WeightedAnDE`
(:mod:`skbn.ande`) and of :class:`~skbn.GaussianAnJE` and
:class:`~skbn.GaussianALR` (:mod:`skbn.anje`), which pool the same factors
differently.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from collections.abc import Sequence
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
)

from ._core import (
    _VAR_SMOOTHING,
    _AttributeEncoder,
    _ChildEncodingMixin,
    _discretized_children,
    _LabelInputMixin,
    _validate_input,
)
from ._kernels import (
    _MAX_LIGHT,
    DISCRETE,
    GAUSSIAN,
    LOG_ZERO,
    SPnDETables,
    _threads,
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


class _BaseAnDE(_LabelInputMixin, _ChildEncodingMixin, ClassifierMixin, BaseEstimator):
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
        X, y = _validate_input(self, X, y)
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
        X, discrete, gaussian = _discretized_children(self, X, y_int)
        self.backoff_, cat_codes = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=gaussian,
            var_prior_weight=self.var_prior_weight,
        )._fit_encoded(X, y_int, self.classes_, discrete, return_codes=True)
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
            known=(types["categorical"], self.backoff_._categories, cat_codes),
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

    @property
    def feature_types_(self):
        """Indices of the attributes of each type the model uses, under the
        keys "gaussian", "categorical" and "bernoulli"."""
        check_is_fitted(self, attributes=["backoff_"])
        return self.backoff_.feature_types_

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
        return self._jll_per_model(_validate_input(self, X, reset=False), per_factor)

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
