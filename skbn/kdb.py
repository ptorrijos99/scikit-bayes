"""
K-Dependence Bayesian classifier (KDB).

This module implements the KDB classifier of Sahami [1], which relaxes the
independence assumption of Naive Bayes by letting every attribute depend on
the class and on at most ``k`` other attributes. The attributes are added to
the network in decreasing order of their mutual information with the class,
and each one takes as parents the (at most ``k``) previously added attributes
with which it has the highest conditional mutual information given the class.

Mixed data types (continuous, categorical, binary) are supported with the
same strategy as the AnDE family and TAN: continuous attributes are
discretized only to learn the structure and when they act as parents, while
as children they remain continuous and are modelled by Gaussian distributions.

References
----------
.. [1] Sahami, M. (1996). Learning limited dependence Bayesian classifiers.
       In Proceedings of the Second International Conference on Knowledge
       Discovery and Data Mining (KDD-96), 335-338.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from numbers import Integral, Real

import numpy as np
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval, StrOptions

from .tan import _BaseAugmentedNB, _conditional_mutual_info


def _mutual_info_with_class(codes, y_idx, cards, n_classes):
    """Empirical mutual information I(A_i; C) (in nats).

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
    mi : ndarray of shape (n_features,)
    """
    n_samples, n_features = codes.shape
    n_c = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    mi = np.zeros(n_features)
    for i in range(n_features):
        ci = int(cards[i])
        key = y_idx.astype(np.int64) * ci + codes[:, i]
        n_ca = (
            np.bincount(key, minlength=n_classes * ci)
            .reshape(n_classes, ci)
            .astype(np.float64)
        )
        n_a = n_ca.sum(axis=0, keepdims=True)
        # sum_{c,a} P(a,c) log[ P(a,c) / (P(a) P(c)) ]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = n_ca * np.log(n_ca * n_samples / (n_c[:, None] * n_a))
        mi[i] = max(np.sum(terms[n_ca > 0]) / n_samples, 0.0)
    return mi


def _kdb_structure(mi, cmi, k):
    """Structure of the KDB network (Sahami, 1996).

    Attributes are placed in decreasing order of ``mi``; each one takes as
    parents the ``min(k, n_placed)`` already placed attributes with the
    highest ``cmi``. Ties are broken deterministically: the attribute with the
    lowest index is placed first, and among candidate parents the one placed
    first is preferred.

    Parameters
    ----------
    mi : ndarray of shape (n_features,)
        Mutual information of every attribute with the class.
    cmi : ndarray of shape (n_features, n_features)
        Conditional mutual information between attributes given the class.
    k : int
        Maximum number of attribute parents.

    Returns
    -------
    order : ndarray of shape (n_features,)
        Attributes in the order in which they are placed.
    parents : list of ndarray
        Attribute parents of every attribute, by decreasing ``cmi``.
    """
    order = np.argsort(-mi, kind="stable").astype(np.int64)
    parents = [None] * len(order)
    for t, i in enumerate(order):
        placed = order[:t]
        best = np.argsort(-cmi[i, placed], kind="stable")[: min(k, t)]
        parents[i] = placed[best]
    return order, parents


class KDB(_BaseAugmentedNB):
    """
    K-Dependence Bayesian (KDB) classifier.

    KDB [1]_ extends Naive Bayes with dependencies between the attributes:
    each attribute has the class and at most ``k`` other attributes as
    parents.

    .. math::
        P(y, \\mathbf{x}) = P(y) \\prod_{i} P(x_i \\mid y, \\mathbf{x}_{\\Pi(i)}),
        \\qquad |\\Pi(i)| \\leq k

    where :math:`\\Pi(i)` is the set of attribute parents of attribute
    :math:`i`. With ``k=0`` the model is Naive Bayes.

    **Structure.** The attributes are sorted by decreasing empirical mutual
    information with the class :math:`I(A_i; C)` and added to the network in
    that order. Each attribute takes as parents the :math:`\\min(k, t)`
    attributes, among the :math:`t` already added, with the highest empirical
    conditional mutual information :math:`I(A_i; A_j \\mid C)`. Continuous
    attributes are discretized (``n_bins``, ``strategy``) to compute these
    quantities.

    **Parameters.** Every conditional
    :math:`P(x_i \\mid y, \\mathbf{x}_{\\Pi(i)})` with at least one attribute
    parent is estimated as Naive Bayes on the augmented class
    :math:`Y^* = (y, \\mathbf{x}_{\\Pi(i)})`, the strategy used by the AnDE
    family and :class:`TAN`: discrete children use Laplace smoothing with
    ``alpha``, continuous children are Gaussian (shrunk towards the pooled
    variance with ``var_prior_weight``), and continuous parents are
    discretized. When a test instance falls in a state
    :math:`(y, \\mathbf{x}_{\\Pi(i)})` never observed in training, the child
    conditional is the uninformed estimate that Laplace smoothing gives an
    empty cell: uniform for discrete children, the pooled Gaussian of the
    attribute for continuous ones. The class prior and the conditionals of the
    attributes without attribute parents are those of a :class:`MixedNB`
    fitted on the original class.

    Parameters
    ----------
    k : int, default=1
        Maximum number of attribute parents of every attribute, in addition
        to the class. An attribute placed when fewer than ``k`` attributes are
        in the network takes all of them as parents. ``k=0`` gives Naive
        Bayes. :meth:`fit` raises a :class:`ValueError` when the joint states
        of the class and the parents of an attribute exceed a 64-bit index,
        which calls for a smaller ``k`` or fewer bins.

    n_bins : int, default=5
        Number of bins used to discretize continuous attributes for the
        structure learning and when they act as parents. Continuous children
        remain continuous.

    strategy : {'uniform', 'quantile', 'kmeans', 'mdlp'}, default='quantile'
        Strategy used for the discretization of continuous attributes.
        ``'mdlp'`` places the cuts with the supervised entropy
        discretization of Fayyad and Irani (see :class:`MDLPDiscretizer`),
        which chooses the number of bins of each attribute, so ``n_bins`` is
        not used.

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

    mutual_info_ : ndarray of shape (n_features,)
        Empirical mutual information :math:`I(A_i; C)` (in nats) between every
        (discretized) attribute and the class.

    conditional_mutual_info_ : ndarray of shape (n_features, n_features)
        Empirical conditional mutual information :math:`I(A_i; A_j \\mid C)`
        (in nats) between every pair of (discretized) attributes.

    feature_order_ : ndarray of shape (n_features,)
        Attributes in the order in which they were added to the network
        (decreasing ``mutual_info_``).

    parents_ : list of ndarray
        ``parents_[i]`` holds the attribute parents of attribute ``i``, by
        decreasing ``conditional_mutual_info_``; it is empty for the
        attributes that depend only on the class.

    backoff_ : MixedNB
        Naive Bayes on the original class. Provides the class prior, the
        conditionals of the attributes without attribute parents and the
        detected feature types (``feature_types_``).

    See Also
    --------
    MixedNB : Naive Bayes classifier for mixed data.
    TAN : Tree Augmented Naive Bayes.
    AnDE : Averaged n-Dependence Estimators.

    References
    ----------
    .. [1] Sahami, M. (1996). Learning limited dependence Bayesian
           classifiers. In Proceedings of the Second International Conference
           on Knowledge Discovery and Data Mining (KDD-96), 335-338.

    Examples
    --------
    >>> import numpy as np
    >>> from skbn import KDB
    >>> rng = np.random.RandomState(0)
    >>> x0 = rng.randint(0, 2, size=200)
    >>> x1 = rng.randint(0, 2, size=200)
    >>> x2 = rng.randint(0, 2, size=200)
    >>> y = x0 ^ x1 ^ x2  # parity: no pair of attributes is informative
    >>> X = np.column_stack([x0, x1, x2])
    >>> clf = KDB(k=2).fit(X, y)
    >>> [p.tolist() for p in clf.parents_]
    [[], [0], [0, 1]]
    >>> clf.score(X, y)
    1.0
    >>> KDB(k=1).fit(X, y).score(X, y) < 0.7
    True
    """

    _parameter_constraints: dict = {
        "k": [Interval(Integral, 0, None, closed="left")],
        "n_bins": [Interval(Integral, 2, None, closed="left")],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans", "mdlp"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "categorical_features": ["array-like", None],
        "bernoulli_features": ["array-like", None],
        "gaussian_features": ["array-like", None],
        "var_prior_weight": [Interval(Real, 0, None, closed="left")],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        k=1,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        var_prior_weight=1.0,
        n_jobs=None,
    ):
        self.k = k
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
        Fit the KDB classifier according to X, y.

        Learns the network structure from the mutual information between
        attributes and class and the conditional mutual information between
        attributes, and estimates the conditional probability tables by
        counting frequencies.

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
            not isinstance(self.k, numbers.Integral)
            or isinstance(self.k, bool)
            or self.k < 0
        ):
            raise ValueError(f"k must be a non-negative integer, got {self.k!r}")

    def _learn_structure(self, codes, y_idx, cards, n_classes):
        """Order by I(A_i; C), parents by I(A_i; A_j | C)."""
        self.mutual_info_ = _mutual_info_with_class(codes, y_idx, cards, n_classes)
        self.conditional_mutual_info_ = _conditional_mutual_info(
            codes, y_idx, cards, n_classes
        )
        self.feature_order_, self.parents_ = _kdb_structure(
            self.mutual_info_, self.conditional_mutual_info_, self.k
        )
        return self.parents_
