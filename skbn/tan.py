"""
Tree Augmented Naive Bayes (TAN).

This module implements the TAN classifier of Friedman et al. [1], which relaxes
the independence assumption of Naive Bayes by letting every attribute depend
on the class and on at most one other attribute. The dependencies form a tree,
learned with the Chow-Liu procedure on the conditional mutual information
between attributes given the class.

Mixed data types (continuous, categorical, binary) are supported with the
same strategy as the AnDE family: continuous attributes are discretized only
to learn the structure and when they act as parents, while as children they
remain continuous and are modelled by Gaussian distributions.

References
----------
.. [1] Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian network
       classifiers. Machine Learning, 29(2-3), 131-163.
.. [2] Chow, C. K., & Liu, C. N. (1968). Approximating discrete probability
       distributions with dependence trees. IEEE Transactions on Information
       Theory, 14(3), 462-467.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ._core import _AttributeEncoder
from ._kernels import (
    DISCRETE,
    GAUSSIAN,
    IGNORED,
    LOG_ZERO,
    _threads,
    conditional_mutual_info,
    conditional_scores,
    count_conditionals,
    count_states,
)
from .ande import _VAR_SMOOTHING, _ChildEncodingMixin
from .mixed_nb import MixedNB


def _conditional_mutual_info(codes, y_idx, cards, n_classes):
    """Empirical conditional mutual information I(A_i; A_j | C) (in nats).

    Parameters
    ----------
    codes : ndarray of shape (n_samples, n_features)
        Integer codes of the (discretized) attributes, in ``[0, cards[j])``.
    y_idx : ndarray of shape (n_samples,)
        Class indices in ``[0, n_classes)``.
    cards : ndarray of shape (n_features,)
        Number of values of each attribute.
    n_classes : int
        Number of classes.

    Returns
    -------
    cmi : ndarray of shape (n_features, n_features)
        Symmetric matrix with zeros on the diagonal.
    """
    return conditional_mutual_info(
        np.ascontiguousarray(codes.T, dtype=np.int32),
        np.asarray(y_idx, dtype=np.int64),
        np.asarray(cards, dtype=np.int64),
        n_classes,
    )


def _maximum_spanning_tree(weights, root):
    """Directed maximum weight spanning tree (Prim's algorithm).

    Ties are broken deterministically: the vertex with the lowest index is
    added first, and a vertex keeps the parent that reached its best weight
    first.

    Parameters
    ----------
    weights : ndarray of shape (n_features, n_features)
        Symmetric edge weights.
    root : int
        Root of the tree; edges are directed away from it.

    Returns
    -------
    parents : ndarray of shape (n_features,)
        Parent of every vertex, ``-1`` for the root.
    """
    n_features = weights.shape[0]
    parents = np.full(n_features, -1, dtype=np.int64)
    in_tree = np.zeros(n_features, dtype=bool)
    in_tree[root] = True
    best = weights[root].astype(np.float64)
    best_parent = np.full(n_features, root, dtype=np.int64)
    for _ in range(n_features - 1):
        k = int(np.argmax(np.where(in_tree, -np.inf, best)))
        parents[k] = best_parent[k]
        in_tree[k] = True
        update = ~in_tree & (weights[k] > best)
        best[update] = weights[k][update]
        best_parent[update] = k
    return parents


class _BaseAugmentedNB(_ChildEncodingMixin, ClassifierMixin, BaseEstimator):
    """Naive Bayes whose attributes may also have other attributes as parents
    (TAN, KDB).

    Subclasses learn the attribute parents in ``_learn_structure``. Every
    conditional :math:`P(x_i \\mid y, \\mathbf{x}_{\\Pi(i)})` is counted on the
    augmented class :math:`(y, \\mathbf{x}_{\\Pi(i)})` and stored in the flat
    tables of :mod:`skbn._kernels`; the attributes without attribute parents
    take their conditionals from the class-level Naive Bayes ``backoff_``.
    """

    def _fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self._check_structure_params()
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)

        # --- 1. Class-level model: prior, parentless conditionals, types ---
        self.backoff_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=self.gaussian_features,
            var_prior_weight=self.var_prior_weight,
        ).fit(X, y)
        types = self.backoff_.feature_types_

        # --- 2. Discretize attributes (structure learning and parents), with
        #        the types the children use ---
        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=types["gaussian"],
            discrete_features=types["categorical"] + types["bernoulli"],
        )
        codes = self._encoder.fit_transform(X)
        gauss = types["gaussian"]
        Xg = np.asarray(X[:, gauss], dtype=float) if gauss else np.zeros((len(X), 0))
        self._pooled_mean = Xg.mean(axis=0) if gauss else np.zeros(0)
        self._pooled_var = Xg.var(axis=0) + (
            self.backoff_.estimators_["gaussian"].epsilon_ if gauss else 0.0
        )

        # --- 3. Structure, then the conditionals of every attribute ---
        with _threads(self.n_jobs):
            parents = self._learn_structure(
                codes, y_idx, self._encoder.cards_, n_classes
            )
            self._fit_conditionals(X, y_idx, codes, parents)
        return self

    def _fit_conditionals(self, X, y_idx, codes, parents):
        """Count ``P(x_i | y, x_parents[i])`` for every attribute that is not
        ignored, and copy the parentless ones from ``backoff_``."""
        types = self.backoff_.feature_types_
        n_classes = len(self.classes_)
        kind, _, gidx, cards = self._layout()
        pcards = self._encoder.cards_
        children = np.flatnonzero(kind != IGNORED).astype(np.int64)
        n_factors = len(children)
        n_parents = np.array([len(parents[i]) for i in children], dtype=np.int64)
        width = max(1, int(n_parents.max(initial=0)))
        pa = np.zeros((n_factors, width), dtype=np.int64)
        strides = np.zeros((n_factors, width), dtype=np.int64)
        n_combos = np.ones(n_factors, dtype=np.int64)
        for m, i in enumerate(children):
            # Mixed radix with the first parent most significant
            for q in range(n_parents[m] - 1, -1, -1):
                pa[m, q] = parents[i][q]
                strides[m, q] = n_combos[m]
                if n_classes * float(n_combos[m]) * pcards[pa[m, q]] >= 2**62:
                    raise ValueError(
                        f"Attribute {i} has {n_parents[m]} parents whose joint"
                        " states with the class exceed a 64-bit index; use fewer"
                        " parents or fewer bins."
                    )
                n_combos[m] *= pcards[pa[m, q]]

        xg = np.ascontiguousarray(X[:, types["gaussian"]], dtype=np.float64)
        value_codes = self._value_codes(X)
        codes_t = np.ascontiguousarray(codes.T)
        child_cards = np.where(kind[children] == DISCRETE, cards[children], 0)
        # A discrete child whose table (states x values) would be much larger
        # than the training data keeps only its non-empty cells (n_cells > 0)
        n_states, n_cells = count_states(
            y_idx,
            codes_t,
            pa,
            strides,
            n_combos,
            n_classes,
            value_codes,
            children,
            child_cards,
        )
        state_off = np.concatenate([[0], np.cumsum(n_states)]).astype(np.int64)
        cell_off = np.concatenate([[0], np.cumsum(n_cells)]).astype(np.int64)
        disc_size = np.where(n_cells > 0, n_cells, n_states * child_cards)
        disc_off = np.concatenate([[0], np.cumsum(disc_size)]).astype(np.int64)
        keys, count, disc, cell_keys, mean, var = count_conditionals(
            y_idx,
            codes_t,
            value_codes,
            xg,
            pa,
            strides,
            children,
            n_combos,
            n_classes,
            state_off,
            disc_off,
            cell_off,
            kind,
            gidx,
            cards,
        )

        # Discrete children: log((N(s, v) + alpha) / (N(s) + alpha * card)),
        # a factor at a time; unseen[s] is the value of N(s, v) = 0 for the
        # factors that keep only their non-empty cells
        unseen = np.zeros(len(count) if n_cells.any() else 0)
        with np.errstate(divide="ignore"):
            for m in np.flatnonzero(child_cards):
                o, card = state_off[m], child_cards[m]
                lo, hi = disc_off[m], disc_off[m + 1]
                if n_cells[m]:
                    local = cell_keys[cell_off[m] : cell_off[m + 1]] // card
                    states = slice(o, state_off[m + 1])
                    unseen[states] = np.log(np.zeros(n_states[m]) + self.alpha) - (
                        np.log(count[states] + self.alpha * card)
                    )
                else:
                    local = np.arange(hi - lo) // card
                disc[lo:hi] = np.log(disc[lo:hi] + self.alpha) - np.log(
                    count[o + local] + self.alpha * card
                )
        np.nan_to_num(disc, copy=False, nan=LOG_ZERO, neginf=LOG_ZERO)
        np.nan_to_num(unseen, copy=False, nan=LOG_ZERO, neginf=LOG_ZERO)

        # Gaussian children: the variance of GaussianNB on the attribute (plus
        # var_smoothing times its variance), shrunk towards that variance
        if xg.shape[1]:
            factor = np.repeat(np.arange(n_factors), n_states)
            is_gauss = kind[children[factor]] == GAUSSIAN
            col_var = np.var(xg, axis=0)[gidx[children[factor]]]
            epsilon = _VAR_SMOOTHING * col_var
            nu = self.var_prior_weight
            shrunk = (count * (var + epsilon) + nu * (col_var + epsilon)) / (count + nu)
            var = np.where(is_gauss, shrunk, var)

        # A Gaussian attribute without attribute parents takes the variance of
        # backoff_, smoothed with the largest variance of all the Gaussian
        # attributes (the discrete ones already match it)
        for m in np.flatnonzero(n_parents == 0):
            j = children[m]
            if kind[j] == GAUSSIAN:
                o = state_off[m]
                var[o : o + n_classes] = self.backoff_.estimators_["gaussian"].var_[
                    :, gidx[j]
                ]

        self._state_count = count
        self._tables = (
            pa,
            strides,
            n_parents,
            children,
            n_combos,
            state_off,
            keys,
            disc_off,
            disc,
            cell_off,
            cell_keys,
            unseen,
            mean,
            var,
            kind,
            gidx,
            cards,
        )

    def _conditional(self, i):
        """The conditional of attribute ``i``: its augmented-state keys
        ``class_index * n_parent_combos + parent_key``, their training counts,
        and the log-probabilities of shape (n_states, n_values) of a discrete
        child or the mean and variance of a Gaussian one."""
        _, _, _, children, _, state_off, keys, disc_off, disc = self._tables[:9]
        cell_off, cell_keys, unseen, mean, var, kind, _, cards = self._tables[9:]
        m = int(np.flatnonzero(children == i)[0])
        lo, hi = state_off[m], state_off[m + 1]
        out = {"keys": keys[lo:hi], "count": self._state_count[lo:hi]}
        values = disc[disc_off[m] : disc_off[m + 1]]
        if kind[i] == DISCRETE and cell_off[m + 1] > cell_off[m]:
            # Only the non-empty cells are stored
            log_prob = np.repeat(unseen[lo:hi], cards[i])
            log_prob[cell_keys[cell_off[m] : cell_off[m + 1]]] = values
            out["log_prob"] = log_prob.reshape(-1, cards[i])
        elif kind[i] == DISCRETE:
            out["log_prob"] = values.reshape(-1, cards[i])
        else:
            out["mean"], out["var"] = mean[lo:hi], var[lo:hi]
        return out

    def _joint_log_likelihood(self, X):
        """Computes log P(y, x) for every sample and class.

        Returns
        -------
        jll : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, attributes=["classes_", "parents_"])
        X = validate_data(self, X, reset=False)
        with _threads(self.n_jobs):
            jll = conditional_scores(
                *self._evaluation_inputs(X), *self._tables, len(self.classes_)
            )
        return jll + self.backoff_.class_log_prior_

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class.
        """
        jll = self._joint_log_likelihood(X)
        log_prob = jll - logsumexp(jll, axis=1, keepdims=True)
        return np.nan_to_num(log_prob, nan=-np.log(jll.shape[1]))

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class.
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
        check_is_fitted(self, attributes=["classes_", "parents_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class TAN(_BaseAugmentedNB):
    """
    Tree Augmented Naive Bayes (TAN) classifier.

    TAN [1]_ extends Naive Bayes with a tree of dependencies between the
    attributes: each attribute has the class and at most one other attribute
    as parents.

    .. math::
        P(y, \\mathbf{x}) = P(y) \\, P(x_r \\mid y)
        \\prod_{i \\neq r} P(x_i \\mid y, x_{\\pi(i)})

    where :math:`r` is the root of the tree and :math:`\\pi(i)` the parent of
    attribute :math:`i`.

    **Structure.** The tree is the maximum weight spanning tree (Chow-Liu
    [2]_) of the complete graph whose edge weights are the empirical
    conditional mutual information :math:`I(A_i; A_j \\mid C)`, directed away
    from ``root``. Continuous attributes are discretized (``n_bins``,
    ``strategy``) to compute these weights.

    **Parameters.** Every conditional :math:`P(x_i \\mid y, x_{\\pi(i)})` is
    estimated as Naive Bayes on the augmented class
    :math:`Y^* = (y, x_{\\pi(i)})`, the strategy used by the AnDE family:
    discrete children use Laplace smoothing with ``alpha``, continuous
    children are Gaussian (shrunk towards the pooled variance with
    ``var_prior_weight``), and continuous parents are discretized. When a test
    instance falls in a state :math:`(y, x_{\\pi(i)})` never observed in
    training, the child conditional is the uninformed estimate that Laplace
    smoothing gives an empty cell: uniform for discrete children, the pooled
    Gaussian of the attribute for continuous ones. The class prior and the
    root conditional are those of a :class:`MixedNB` fitted on the original
    class.

    Parameters
    ----------
    root : int, default=0
        Index of the attribute used as root of the tree, in
        ``[0, n_features - 1]``. The choice does not change the learned
        (undirected) tree, only the direction of its edges.

    n_bins : int, default=5
        Number of bins used to discretize continuous attributes for the
        structure learning and when they act as parents. Continuous children
        remain continuous.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for the discretization of continuous attributes.

    alpha : float, default=1.0
        Non-negative additive (Laplace/Lidstone) smoothing parameter for the
        discrete conditional probability tables.

    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the features to treat as categorical,
        passed to :class:`MixedNB`, which describes how the undeclared
        features are typed.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the features to treat as Bernoulli.
        See :class:`MixedNB`.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the features to treat as Gaussian.
        See :class:`MixedNB`.

    var_prior_weight : float, default=1.0
        Non-negative number of pseudo-observations shrinking each Gaussian
        variance towards the pooled variance of the feature (see
        :class:`MixedNB`).

    n_jobs : int, default=None
        Number of threads used to compute the conditional mutual information
        and to count and evaluate the conditionals. ``None`` means 1 and
        ``-1`` all the processors.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    conditional_mutual_info_ : ndarray of shape (n_features, n_features)
        Empirical conditional mutual information :math:`I(A_i; A_j \\mid C)`
        (in nats) between every pair of (discretized) attributes.

    parents_ : ndarray of shape (n_features,)
        Attribute parent of every attribute in the tree, ``-1`` for the root.

    backoff_ : MixedNB
        Naive Bayes on the original class. Provides the class prior, the root
        conditional and the detected feature types (``feature_types_``).

    See Also
    --------
    MixedNB : Naive Bayes classifier for mixed data.
    AnDE : Averaged n-Dependence Estimators.

    References
    ----------
    .. [1] Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian network
           classifiers. Machine Learning, 29(2-3), 131-163.
    .. [2] Chow, C. K., & Liu, C. N. (1968). Approximating discrete probability
           distributions with dependence trees. IEEE Transactions on
           Information Theory, 14(3), 462-467.

    Examples
    --------
    >>> import numpy as np
    >>> from skbn import TAN
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randint(0, 2, size=200)
    >>> x0 = rng.randint(0, 3, size=200)
    >>> x1 = (x0 + y) % 3  # depends on x0 and the class
    >>> x2 = rng.randint(0, 3, size=200)  # noise
    >>> X = np.column_stack([x0, x1, x2])
    >>> clf = TAN().fit(X, y)
    >>> clf.parents_.tolist()
    [-1, 0, 0]
    >>> clf.score(X, y)
    1.0
    """

    _parameter_constraints: dict = {
        "root": [Interval(Integral, 0, None, closed="left")],
        "n_bins": [Interval(Integral, 2, None, closed="left")],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "categorical_features": ["array-like", None],
        "bernoulli_features": ["array-like", None],
        "gaussian_features": ["array-like", None],
        "var_prior_weight": [Interval(Real, 0, None, closed="left")],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        root=0,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        var_prior_weight=1.0,
        n_jobs=None,
    ):
        self.root = root
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features
        self.var_prior_weight = var_prior_weight
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the TAN classifier according to X, y.

        Learns the tree structure from the conditional mutual information
        between attributes and estimates the conditional probability tables
        by counting frequencies.

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
        return self._fit(X, y)

    def _check_structure_params(self):
        if (
            not isinstance(self.root, numbers.Integral)
            or isinstance(self.root, bool)
            or not 0 <= self.root < self.n_features_in_
        ):
            raise ValueError(
                f"root must be an integer in [0, {self.n_features_in_ - 1}], got"
                f" {self.root!r}"
            )

    def _learn_structure(self, codes, y_idx, cards, n_classes):
        """Chow-Liu tree on I(A_i; A_j | C), directed away from ``root``."""
        self.conditional_mutual_info_ = _conditional_mutual_info(
            codes, y_idx, cards, n_classes
        )
        self.parents_ = _maximum_spanning_tree(self.conditional_mutual_info_, self.root)
        return [[] if p < 0 else [p] for p in self.parents_]
