"""
Averaged n-Join Estimators (AnJE) and Accelerated Logistic Regression (ALR).

Both models follow Zaidi et al. [1]_. An n-join is the joint distribution
:math:`P(x_\\alpha | y)` of a subset :math:`\\alpha` of ``n`` attributes;
AnJE pools every n-join with a fixed weight and ALR learns one weight per
n-join, value and class by maximising the conditional log-likelihood. They are
defined on discrete attributes, so :class:`AnJE` and :class:`ALR` discretize
continuous attributes.

By the chain rule an (n+1)-join is the product of the factors of an SPnDE,
:math:`P(x_\\alpha | y) = P(x_P | y) P(x_j | y, x_P)` for every
:math:`j \\in \\alpha` and :math:`P = \\alpha \\setminus \\{j\\}`, so both models
are log-linear pools of the factors of the SPnDEs of order n.
:class:`GaussianAnJE` and :class:`GaussianALR` build them on those factors,
which extends them to continuous attributes as HAODE extends AODE: the
super-parents are discretized and every child keeps the conditional family of
its type, Gaussian for continuous attributes.

References
----------
.. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., & Cerquides, J. (2016).
       ALR^n: Accelerated higher-order logistic regression. Machine Learning, 104(2-3), 151-194.
.. [2] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
       Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
       Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from itertools import combinations
from math import comb
from numbers import Integral, Real

import numpy as np
from numba import get_num_threads
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ._core import (
    _AttributeEncoder,
    _feature_indices,
    _fit_weights,
    _row_chunks,
    _warn_not_converged,
)
from ._kernels import (
    _threads,
    join_cells_seen,
    join_counts,
    join_key_counts,
    join_keys,
    join_objective,
    join_rows,
    join_scores,
)
from .ande import WeightedAnDE, _BaseAnDE, _geometric_scores

# Key spaces of the n-joins up to this size map keys to table rows with a
# dense array (4 bytes per key); larger ones use a binary search
_MAX_DENSE_KEYS = 1 << 26

# Key spaces with at most this many keys times classes are counted in one
# pass, into a dense array (4 bytes per key and class)
_MAX_KEY_COUNTS = 1 << 24

# ALR keeps the table rows of the training samples (4 bytes per sample and
# join) when there are at most this many of them
_MAX_CACHED_ROWS = 1 << 27

# L-BFGS-B settings of the code of the ALR experiments (Zaidi et al., 2016):
# factr = 1e7 (a relative reduction of the objective of 1e7 machine epsilons),
# a projected-gradient tolerance of 1e-33 and 5 corrections
_ZAIDI_FTOL = 1e7 * np.finfo(float).eps
_ZAIDI_GTOL = 1e-33
_ZAIDI_MAXCOR = 5


class AnJE(ClassifierMixin, BaseEstimator):
    """
    Averaged n-Join Estimators (AnJE) [Generative].

    The geometric mean of the n-join estimators of every subset of ``n`` of the
    :math:`a` attributes (Zaidi et al. [1]_, Eq. 11):

    .. math::
        P(y | x) \\propto P(y) \\prod_{\\alpha \\in \\binom{A}{n}}
        P(x_\\alpha | y)^{\\frac{(n-1)!(a-n)!}{(a-1)!}}.

    The exponent is the inverse of the number of subsets that contain a given
    attribute, so that every attribute enters the product with total exponent
    one. A1JE is Naive Bayes, and AnJE uses the same statistics as A(n-1)DE.
    Probabilities are m-estimates with ``m = alpha``:
    :math:`P(y) = (\\#y + m/|Y|) / (t + m)` and
    :math:`P(x_\\alpha | y) = (\\#(x_\\alpha, y) + m/|X_\\alpha|) / (\\#y + m)`.

    Continuous attributes are discretized into ``n_bins`` bins (the original
    work uses MDL discretization, available as :class:`MDLPDiscretizer` in a
    pipeline, or ``strategy='mdlp'``). A discrete value never seen in training
    carries no information about the class: the n-joins that contain it are
    left out for that sample, as a missing value would be.

    Parameters
    ----------
    n_join : int, default=2
        Size ``n`` of the attribute subsets (at least 1). With fewer attributes
        than ``n_join``, the single join of all attributes is used.

    n_bins : int, default=5
        Number of bins for continuous attributes (at least 2).

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Binning strategy for continuous attributes.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Non-negative weight ``m`` of the m-estimates; with ``0`` the cells
        unseen in training have zero probability.

    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as discrete
        regardless of their dtype.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as discrete
        regardless of their dtype.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as continuous
        (binned) regardless of their dtype.
        The attributes in none of the three parameters are typed from their
        training values with the rules of :class:`MixedNB`, and only those
        typed as Gaussian are binned.

    n_jobs : int, default=None
        Number of threads used to count and evaluate the n-joins.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    joins_ : ndarray of shape (n_joins, n)
        Attribute subsets of the n-joins, one row per subset, with
        ``n = min(n_join, n_features_in_)``.

    See Also
    --------
    ALR : AnJE with discriminatively learned weights.
    GaussianAnJE : AnJE on the factors of the SPnDEs, with Gaussian children.
    AnDE : Averaged n-Dependence Estimators.

    References
    ----------
    .. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., &
           Cerquides, J. (2016). ALR^n: Accelerated higher-order logistic
           regression. Machine Learning, 104(2-3), 151-194.
    """

    _parameter_constraints: dict = {
        "n_join": [Interval(Integral, 1, None, closed="left")],
        "n_bins": [Interval(Integral, 2, None, closed="left")],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans", "mdlp"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "categorical_features": ["array-like", None],
        "bernoulli_features": ["array-like", None],
        "gaussian_features": ["array-like", None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        n_join=2,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        n_jobs=None,
    ):
        self.n_join = n_join
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Count the n-joins of the training data.

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
        self._fit_generative(X, y)
        return self

    def _fit_generative(self, X, y):
        """Fit the tables and return the transposed training codes and the
        class indices."""
        X, y = validate_data(self, X, y)
        if not isinstance(self.n_join, numbers.Integral) or self.n_join < 1:
            raise ValueError(f"n_join must be a positive integer, got {self.n_join!r}")
        self.classes_ = unique_labels(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=_feature_indices(
                self.gaussian_features, n_features, "gaussian_features"
            ),
            discrete_features=_feature_indices(
                self.categorical_features, n_features, "categorical_features"
            )
            + _feature_indices(
                self.bernoulli_features, n_features, "bernoulli_features"
            ),
        )
        codes = self._encoder.fit_transform(X, y_idx, dtype=np.int32)

        # --- Cell keys: 0 is the class prior, then one block per join ---
        n = min(self.n_join, n_features)
        self.joins_ = np.array(list(combinations(range(n_features), n)), dtype=np.int64)
        # Each attribute gets an extra code for values unseen in training
        slots = (self._encoder.cards_ + 1)[self.joins_]
        sizes = slots.prod(axis=1)
        # Row-major strides of the cells of each join
        self._join_strides = np.ones_like(slots)
        self._join_strides[:, :-1] = np.cumprod(slots[:, :0:-1], axis=1)[:, ::-1]
        self._join_offsets = 1 + np.cumsum(sizes) - sizes
        self._join_weight = 1.0 / comb(n_features - 1, n - 1)  # Eq. 10

        # --- Table rows: the cells seen in training, then one empty cell per
        # join that stands for all its unseen cells (they share its estimate) ---
        if slots.astype(float).prod(axis=1).sum() >= 2**62:
            raise ValueError(
                "The cells of the n-joins exceed a 64-bit index; reduce n_join or"
                " n_bins."
            )
        n_keys = 1 + int(sizes.sum())
        codes_t = np.ascontiguousarray(codes.T, dtype=np.int32)
        joins = (self.joins_, self._join_strides, self._join_offsets)
        n_joins = len(self.joins_)
        with _threads(self.n_jobs):
            key_counts = None
            if n_keys * n_classes <= _MAX_KEY_COUNTS:
                # The counts of every key in one pass give the cells seen
                key_counts = join_key_counts(codes_t, y_idx, *joins, n_keys, n_classes)
                self._cell_keys = np.flatnonzero(key_counts.any(axis=1))
            elif n_keys <= _MAX_DENSE_KEYS:
                self._cell_keys = np.flatnonzero(
                    join_cells_seen(codes_t, *joins, n_keys)
                )
            # A dense map from keys to rows when it fits, else a binary search
            if n_keys <= _MAX_DENSE_KEYS:
                n_seen = len(self._cell_keys)
                self._row_of_cell = np.concatenate(
                    [[0], np.repeat(n_seen + np.arange(n_joins), sizes)]
                ).astype(np.int32)
                self._row_of_cell[self._cell_keys] = np.arange(n_seen)
            else:
                keys = join_keys(codes_t, *joins)
                self._cell_keys = np.unique(np.append(keys, 0))
                self._row_of_cell = np.zeros(0, dtype=np.int32)
            self._n_rows = len(self._cell_keys) + n_joins
            self._row_join = np.concatenate(  # -1 for the prior
                [
                    np.searchsorted(self._join_offsets, self._cell_keys, side="right")
                    - 1,
                    np.arange(n_joins),
                ]
            )

            # --- m-estimates (row 0 counts the classes: every sample is in it) ---
            if key_counts is None:
                counts = join_counts(codes_t, y_idx, *self._cell_args(), n_classes)
            else:
                counts = np.zeros((self._n_rows, n_classes))
                counts[: len(self._cell_keys)] = key_counts[self._cell_keys]
        class_counts = counts[0]
        m = self.alpha
        n_cells = self._encoder.cards_[self.joins_].prod(axis=1)  # |X_alpha|
        cell_prior = (m / n_cells)[self._row_join[1:]][:, None]
        with np.errstate(divide="ignore"):
            self._log_table = np.vstack(
                [
                    np.log((class_counts + m / n_classes) / (n_samples + m)),
                    np.log(counts[1:] + cell_prior) - np.log(class_counts + m),
                ]
            )
        return codes_t, y_idx

    def _cell_args(self):
        """How the kernels of :mod:`skbn._kernels` find the table row of a
        cell: the joins, their strides and key offsets, and the row map."""
        return (
            self.joins_,
            self._join_strides,
            self._join_offsets,
            self._row_of_cell,
            self._cell_keys,
        )

    def _transform_codes(self, codes, unseen):
        """Transposed codes, shape (n_features, n_samples), from the output
        of the encoder; an unseen value takes the extra code of its
        attribute, so that its cells are those of no training sample."""
        codes = np.where(unseen, self._encoder.cards_, codes)
        return np.ascontiguousarray(codes.T, dtype=np.int32)

    def _drop_unseen_joins(self, scores, codes_t, unseen, table):
        """Rescore, in place, the samples that hold a value never seen in
        training without the n-joins that contain it.

        Such a join would add the estimate of an empty cell,
        ``(m / |X_alpha|) / (#y + m)``, which says nothing about the value and
        favours the classes with few samples. Only those samples are
        rescored, so data without unseen values cost nothing.
        """
        rows = np.flatnonzero(unseen.any(axis=1))
        if not len(rows):
            return
        n_joins, n_classes = len(self.joins_), table.shape[1]
        # A last row of zeros stands for the joins that are left out
        padded = np.vstack([table, np.zeros((1, n_classes))])
        for chunk in _row_chunks(len(rows), n_joins * (n_classes + 1)):
            r = rows[chunk]
            cells = join_rows(
                np.ascontiguousarray(codes_t[:, r]), *self._cell_args(), True
            )
            cells[unseen[r][:, self.joins_].any(axis=2)] = len(table)
            scores[r] = table[0] + padded[cells].sum(axis=1)

    def _table(self):
        """Log-table weighted as in the AnJE: the joins by Eq. 10, the prior by 1."""
        table = self._log_table * self._join_weight
        table[0] = self._log_table[0]
        return table

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
            ``classes_``.
        """
        check_is_fitted(self, attributes=["classes_", "joins_"])
        X = validate_data(self, X, reset=False)
        codes, unseen = self._encoder.transform(X)
        codes_t = self._transform_codes(codes, unseen)
        table = self._table()
        with _threads(self.n_jobs):
            scores = join_scores(codes_t, *self._cell_args(), table)
            self._drop_unseen_joins(scores, codes_t, unseen, table)
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        return np.nan_to_num(log_prob, nan=-np.log(scores.shape[1]))

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
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class ALR(AnJE):
    """
    Accelerated Higher-Order Logistic Regression (ALR^n) [Hybrid].

    The model of Zaidi et al. [1]_ (Eq. 13): every n-join probability of the
    AnJE gets a weight, learned by maximising the conditional log-likelihood,

    .. math::
        \\log P(y | x) = w_y \\log P(y) + \\sum_{\\alpha \\in \\binom{A}{n}}
        w_{y, \\alpha, x_\\alpha} \\log P(x_\\alpha | y) - \\log Z(x).

    The objective is convex. Weights start at 1 and are optimized with
    L-BFGS-B. The defaults reproduce the settings of the code of the original
    experiments [1]_ [2]_: no penalty, 5 L-BFGS-B corrections, no stopping test
    on the projected gradient (its tolerance is 1e-33), and the optimizer
    stops when the relative reduction of the objective falls below 1e7
    machine epsilons (L-BFGS-B's ``factr = 1e7``) or after 10^4 iterations.
    ``l2_reg`` adds the penalty towards 1 (the generative estimates) that [1]_
    suggests for small datasets but does not evaluate.

    ``weight_level=3`` is the ALR^n of the paper. Levels 1 and 2 tie the
    weights across classes and values, following the nested
    parameterizations of Zaidi et al. [2]_ (Sec. 5.4), which the ALR paper
    does not use:

    ===== ==============================================
    Level One weight per
    ===== ==============================================
    1     n-join :math:`\\alpha`
    2     n-join and value :math:`(\\alpha, x_\\alpha)`
    3     n-join, value and class (ALR^n)
    ===== ==============================================

    The class prior is weighted like an n-join with a single value. Only the
    cells observed in training get their own weight at levels 2 and 3; the
    unseen cells of an n-join share one weight that is never trained and keeps
    its value of 1, so the number of weights is bounded by the data.

    Parameters
    ----------
    n_join : int, default=2
        Size ``n`` of the attribute subsets (at least 1; ALR2 uses pairs).
        With fewer attributes than ``n_join``, the single join of all
        attributes is used.

    n_bins : int, default=5
        Number of bins for continuous attributes (at least 2).

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Binning strategy for continuous attributes.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Weight ``m`` of the m-estimates; must be positive.

    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as discrete
        regardless of their dtype.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as discrete
        regardless of their dtype.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the attributes to treat as continuous
        (binned) regardless of their dtype.
        The attributes in none of the three parameters are typed from their
        training values with the rules of :class:`MixedNB`, and only those
        typed as Gaussian are binned.

    weight_level : int, default=3
        Granularity of the weights, 1, 2 or 3 (see the table above).

    l2_reg : float, default=0.0
        Non-negative L2 regularization towards the generative weights
        (``w = 1``), added to the summed conditional log-likelihood.

    max_iter : int, default=10000
        Maximum number of L-BFGS-B iterations (at least 1). A
        :class:`~sklearn.exceptions.ConvergenceWarning` is raised at the end of
        :meth:`fit` when L-BFGS-B stops at ``max_iter`` before converging;
        increase ``max_iter`` or ``tol`` in that case.

    tol : float or None, default=2.22e-9
        Stopping tolerance on the relative reduction of the objective,
        ``(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= tol``; the default is
        1e7 machine epsilons, as in the original code, and None uses SciPy's
        default (the same value).

    n_jobs : int, default=None
        Number of threads used to count the n-joins and to evaluate the
        objective. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    joins_ : ndarray of shape (n_joins, n)
        Attribute subsets of the n-joins, one row per subset, with
        ``n = min(n_join, n_features_in_)``.

    learned_weights_ : ndarray of shape (n_weights,)
        Learned weights, one row per weight group and, at level 3, one column
        per class, flattened in row-major order.

    n_weights_ : int
        Number of weights.

    n_iter_ : int
        Number of L-BFGS-B iterations.

    See Also
    --------
    AnJE : The generative model on the same n-joins. ``w = 1`` is the
        unaveraged product of the n-joins, not the AnJE.
    GaussianALR : ALR^n on the factors of the SPnDEs, with Gaussian children.
    WeightedAnDE : Discriminative weights on the AnDE ensemble.

    References
    ----------
    .. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., &
           Cerquides, J. (2016). ALR^n: Accelerated higher-order logistic
           regression. Machine Learning, 104(2-3), 151-194.
    .. [2] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
           Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
           Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.
    """

    _parameter_constraints: dict = {
        **AnJE._parameter_constraints,
        "weight_level": [Interval(Integral, 1, 3, closed="both")],
        "l2_reg": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left"), None],
    }

    def __init__(
        self,
        n_join=2,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        weight_level=3,
        l2_reg=0.0,
        max_iter=10000,
        tol=_ZAIDI_FTOL,
        n_jobs=None,
    ):
        super().__init__(
            n_join,
            n_bins,
            strategy,
            alpha,
            categorical_features,
            bernoulli_features,
            gaussian_features,
            n_jobs,
        )
        self.weight_level = weight_level
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the AnJE tables, then the discriminative weights.

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
        if self.weight_level not in (1, 2, 3):
            raise ValueError(f"weight_level must be 1, 2 or 3, got {self.weight_level}")
        if not self.alpha > 0:
            raise ValueError(
                "ALR needs alpha > 0: an empty cell would have log-probability -inf"
            )
        codes_t, y_idx = self._fit_generative(X, y)
        n_classes = len(self.classes_)

        # Weight group of every table row
        if self.weight_level == 1:
            self._groups = 1 + self._row_join  # the prior row has join -1
        else:
            self._groups = np.arange(self._n_rows)
        self._n_columns = n_classes if self.weight_level == 3 else 1
        self.n_weights_ = (int(self._groups.max()) + 1) * self._n_columns

        with _threads(self.n_jobs):
            # The table rows of the training samples, computed once when they fit
            # in the budget (the objective is evaluated thousands of times)
            if codes_t.shape[1] * len(self.joins_) <= _MAX_CACHED_ROWS:
                rows = join_rows(codes_t, *self._cell_args(), n_classes == 2)
            else:
                rows = np.zeros((0, 0), dtype=np.int32)
            self.learned_weights_, self.n_iter_, converged = _fit_weights(
                lambda theta: self._nll_and_grad(theta, codes_t, y_idx, rows),
                np.ones(self.n_weights_),
                self.l2_reg,
                self.max_iter,
                tol=self.tol,
                gtol=_ZAIDI_GTOL,
                maxcor=_ZAIDI_MAXCOR,
            )
        _warn_not_converged(not converged, 1, self.max_iter)
        return self

    def _nll_and_grad(self, theta, codes_t, y_idx, rows=None):
        """Conditional negative log-likelihood and its gradient, on the
        transposed codes of the samples and their class indices (``rows``
        optionally holds their table rows, from :func:`join_rows`)."""
        if rows is None:
            rows = np.zeros((0, 0), dtype=np.int32)
        weights = theta.reshape(-1, self._n_columns)
        # Above level 1 every table row has its own weight group
        own = self.weight_level > 1
        nll, dtable = join_objective(
            codes_t,
            y_idx,
            *self._cell_args(),
            rows,
            (weights if own else weights[self._groups]) * self._log_table,
            get_num_threads(),
        )
        # dNLL/dweight of a row: its log-probability times the residuals of
        # the samples that fall in it
        drow = dtable * self._log_table
        if self._n_columns == 1:
            drow = drow.sum(axis=1, keepdims=True)
        if own:
            return nll, drow.ravel()
        grad = np.column_stack(
            [
                np.bincount(self._groups, weights=col, minlength=len(weights))
                for col in drow.T
            ]
        )
        return nll, grad.ravel()

    def _table(self):
        weights = self.learned_weights_.reshape(-1, self._n_columns)[self._groups]
        return weights * self._log_table

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
            ``classes_``, from the weighted n-joins.
        """
        check_is_fitted(self, attributes=["classes_", "joins_", "learned_weights_"])
        return super().predict_log_proba(X)


# =============================================================================
# AnJE and ALR on the factors of the SPnDEs (Gaussian children)
# =============================================================================


def _join_exponents(n_features, n_dependence):
    """Exponents of the root and child factors of an SPnDE that turn the mean
    of the components' log-factors into the AnJE with joins of size n + 1.

    Averaging the n + 1 orders of the chain rule of every join and pooling the
    joins with the AnJE exponent ``1 / C(d-1, n)`` gives
    ``mean_P [root log P(x_P | y) + child sum_j log P(x_j | y, x_P)]`` with
    ``root = d / (n+1)`` and ``child = d / ((n+1)(d-n))``. With ``d = n`` the
    single component has no children and its root is the join of all the
    attributes.
    """
    d, n = n_features, n_dependence
    if d <= n:
        return 1.0, 0.0
    return d / (n + 1), d / ((n + 1) * (d - n))


def _root_log_marginals(model):
    """Class marginal ``log sum_{x_P} P(y, x_P)`` of the root factor of every
    SPnDE of a fitted ensemble, shape (n_classes, n_models).

    Laplace smoothing over the grid of augmented states gives class ``y`` the
    mass ``(#y + alpha K) / (t + alpha |Y| K)``, with ``K`` the number of parent
    instantiations, which is not ``P(y)``. Dividing the root by it gives the
    conditional ``P(x_P | y)`` of the join. Without parents the root is the
    class prior itself, and dividing by it leaves ``P(x_P | y) = 1``.
    """
    if model.n_dependence == 0:
        return model.backoff_.class_log_prior_[:, None]
    counts = model.class_count_[:, None]
    combos = np.array([info["n_parent_combos"] for info in model.ensemble_])
    return np.log(counts + model.alpha * combos) - np.log(
        counts.sum() + model.alpha * len(model.classes_) * combos
    )


class GaussianAnJE(_BaseAnDE):
    """
    Averaged (n+1)-Join Estimator on the factors of the SPnDEs [Generative].

    The AnJE of Zaidi et al. [1]_ with joins of size ``n_dependence + 1``,
    written as a log-linear pool of the factors of the SPnDEs of order
    ``n_dependence``:

    .. math::
        \\log P(y | x) \\propto \\log P(y) + \\frac{1}{|M|} \\sum_{P \\in M}
        \\Big[ r \\log P(x_P | y) + c \\sum_{j \\notin P} \\log P(x_j | y, x_P)
        \\Big],

    with :math:`r = d / (n+1)` and :math:`c = d / ((n+1)(d-n))`, where
    :math:`M` holds the super-parent sets of the components that take part in
    the prediction. :class:`AnDE` with ``mean="geometric"`` pools the same
    factors with exponents :math:`1/|M|`.

    The factors are those of :class:`AnDE`, so the model accepts the same
    mixed data: super-parents are discretized into ``n_bins`` bins and every
    child is scored by the conditional family of its type, Gaussian for
    continuous attributes, as in HAODE. The root
    :math:`P(x_P | y)` is the Laplace-smoothed probability of the augmented
    state divided by its total within the class, so that every factor is a
    distribution conditional on the class. On discrete data the model differs
    from :class:`AnJE` with ``n_join=n_dependence + 1`` only in the smoothing,
    Laplace on the augmented states instead of m-estimates on the joins, and
    the two coincide as ``alpha`` tends to zero. ``n_dependence=0`` gives
    Naive Bayes, which is A1JE.

    Parameters
    ----------
    n_dependence : int, default=1
        Order ``n`` of the SPnDEs; the joins have ``n + 1`` attributes. It must
        satisfy ``0 <= n_dependence <= n_features``; ``0`` gives Naive Bayes.

    n_bins : int, default=5
        Number of bins of the continuous super-parents (at least 2).

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Binning strategy of the continuous super-parents.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Non-negative Laplace smoothing of the discrete factors and of the
        augmented-state prior.

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
        pooled variance of the attribute (see :class:`MixedNB`).

    min_parent_count : int, default=1
        Minimum training frequency of a parent instantiation for its SPnDE to
        take part in the prediction of an instance. Values below 1 behave as 1.

    unseen_state : {"laplace", "backoff"}, default="laplace"
        Child conditionals of an augmented state never seen in training (see
        :class:`AnDE`).

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
        Naive Bayes on the original class, used for the class prior, the
        class-level conditionals and as the fallback when no SPnDE qualifies.

    See Also
    --------
    AnJE : The AnJE on discretized attributes, with the m-estimates of [1]_.
    GaussianALR : Discriminative weights on the factors of this model.

    References
    ----------
    .. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., &
           Cerquides, J. (2016). ALR^n: Accelerated higher-order logistic
           regression. Machine Learning, 104(2-3), 151-194.
    """

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
        root, child = _join_exponents(self.n_features_in_, self.n_dependence)
        log_marginals = _root_log_marginals(self)
        n_classes, n_models = len(self.classes_), len(self.ensemble_)
        n_terms = 1 + self.n_features_in_ - self.n_dependence
        out = np.empty((X.shape[0], n_classes))
        for rows in _row_chunks(X.shape[0], n_classes * n_models * n_terms):
            ll, _, active, nb_jll = self._jll_per_model(X[rows], per_factor=True)
            jll = root * (ll[..., 0] - log_marginals) + child * ll[..., 1:].sum(axis=3)
            scores = _geometric_scores(jll, active, nb_jll)
            scores[active.any(axis=1)] += self.backoff_.class_log_prior_
            out[rows] = self._normalize(scores)
        return out


class GaussianALR(WeightedAnDE):
    """
    ALR^n on the factors of the SPnDEs [Hybrid].

    The ALR^n of Zaidi et al. [1]_ with joins of size ``n_dependence + 1``,
    built on the factors of :class:`GaussianAnJE`: every factor gets one
    weight per value, class and parent instantiation (the factor weights of
    :class:`WeightedAnDE` at level 4), the weighted factors are pooled with
    the AnJE exponents, and the weights are learned by maximising the
    conditional log-likelihood,

    .. math::
        \\log P(y | x) = \\log P(y) + \\frac{1}{|M|} \\sum_{P \\in M}
        \\Big[ r\\, w_{y, x_P} \\log P(x_P | y) + c \\sum_{j \\notin P}
        w_{j, x_j, y, x_P} \\log P(x_j | y, x_P) \\Big] - \\log Z(x).

    At ``w = 1`` the model is :class:`GaussianAnJE`. The objective is convex.
    On discrete data the model spans the same family as :class:`ALR` with
    ``n_join=n_dependence + 1``, logistic regression with (n+1)-way
    interactions, with a different parameterization of the weights. The
    defaults reproduce the L-BFGS-B settings of the code of the ALR^n
    experiments [1]_, as in :class:`ALR`: no penalty, 5 corrections, no
    stopping test on the projected gradient, and a stop when the relative
    reduction of the objective falls below 1e7 machine epsilons or after 10^4
    iterations. Only the combinations that occur in training get their own
    weight, as in :class:`WeightedAnDE`.

    Parameters
    ----------
    n_dependence : int, default=1
        Order ``n`` of the SPnDEs; the joins have ``n + 1`` attributes. It must
        satisfy ``0 <= n_dependence <= n_features``; ``0`` gives Naive Bayes.

    n_bins : int, default=5
        Number of bins of the continuous super-parents (at least 2).

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Binning strategy of the continuous super-parents.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

    alpha : float, default=1.0
        Laplace smoothing of the discrete factors and of the augmented-state
        prior; must be positive.

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
        pooled variance of the attribute (see :class:`MixedNB`).

    min_parent_count : int, default=1
        Minimum training frequency of a parent instantiation for its SPnDE to
        take part in the prediction of an instance. Values below 1 behave as 1.

    unseen_state : {"laplace", "backoff"}, default="laplace"
        Child conditionals of an augmented state never seen in training (see
        :class:`AnDE`).

    l2_reg : float, default=0.0
        Non-negative L2 regularization towards the generative weights
        (``w = 1``), added to the summed conditional log-likelihood.

    max_iter : int, default=10000
        Maximum number of L-BFGS-B iterations (at least 1). A
        :class:`~sklearn.exceptions.ConvergenceWarning` is raised at the end of
        :meth:`fit` when L-BFGS-B stops at ``max_iter`` before converging;
        increase ``max_iter`` or ``tol`` in that case.

    tol : float or None, default=2.22e-9
        Stopping tolerance on the relative reduction of the objective,
        ``(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= tol``; the default is
        1e7 machine epsilons, as in the original code, and None uses SciPy's
        default (the same value).

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
        Naive Bayes on the original class, used for the class prior, the
        class-level conditionals and as the fallback when no SPnDE qualifies.

    coef_theta_ : ndarray of shape (n_weights,)
        Optimized parameters, equal to ``learned_weights_`` since the weights
        are exponents.

    learned_weights_ : ndarray of shape (n_weights,)
        Learned weights ``w``.

    n_weights_ : int
        Number of weights, including the shared one for unseen combinations.

    n_iter_ : int
        Number of L-BFGS-B iterations.

    See Also
    --------
    ALR : The ALR^n on discretized attributes, with the m-estimates of [1]_.
    GaussianAnJE : The generative model at ``w = 1``.

    References
    ----------
    .. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., &
           Cerquides, J. (2016). ALR^n: Accelerated higher-order logistic
           regression. Machine Learning, 104(2-3), 151-194.
    """

    # The configuration of WeightedAnDE that this model fixes
    mean = "geometric"
    average = "joint"
    weight_scope = "factor"
    weight_level = 4
    modular = False
    _gtol = _ZAIDI_GTOL
    _maxcor = _ZAIDI_MAXCOR

    _parameter_constraints: dict = {
        **_BaseAnDE._parameter_constraints,
        "l2_reg": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left"), None],
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
        l2_reg=0.0,
        max_iter=10000,
        tol=_ZAIDI_FTOL,
    ):
        _BaseAnDE.__init__(
            self,
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
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the factors of the SPnDEs, then the discriminative weights.

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
        if not self.alpha > 0:
            raise ValueError(
                "GaussianALR needs alpha > 0: an unseen augmented state would"
                " have log-probability -inf"
            )
        return super().fit(X, y)

    def _design(self, X):
        """Factor-weight design with the root divided by its class marginal,
        the AnJE exponents folded into the log-terms and the class prior as
        intercept."""
        design, nb_jll = super()._design(X)
        root, child = _join_exponents(self.n_features_in_, self.n_dependence)
        n_samples, n_models = design.active.shape
        n_classes = len(self.classes_)
        terms = design.operator.data.reshape(n_samples, n_classes, n_models, -1)
        terms[..., 0] -= _root_log_marginals(self)
        terms[..., 0] *= root
        terms[..., 1:] *= child
        intercept = np.broadcast_to(
            self.backoff_.class_log_prior_[None, :, None],
            (n_samples, n_classes, n_models),
        )
        return design._replace(intercept=intercept), nb_jll
