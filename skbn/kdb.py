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

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ._core import _AttributeEncoder
from .ande import _lookup
from .mixed_nb import MixedNB
from .tan import _conditional_mutual_info


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


def _augmented_keys(class_idx, parent_codes, n_classes, parent_cards):
    """Linear index of the augmented class ``(y, x_pa)`` in mixed radix.

    ``class_idx`` and every array in ``parent_codes`` are broadcast together.
    """
    return np.ravel_multi_index((class_idx, *parent_codes), (n_classes, *parent_cards))


class KDB(ClassifierMixin, BaseEstimator):
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
    parent is estimated by a :class:`MixedNB` fitted on the augmented class
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
        Bayes.

    n_bins : int, default=5
        Number of bins used to discretize continuous attributes for the
        structure learning and when they act as parents. Continuous children
        remain continuous.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for the discretization of continuous attributes.

    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter for the discrete
        conditional probability tables.

    categorical_features : array-like of int, default=None
        Indices of the features to treat as categorical. See :class:`MixedNB`.

    bernoulli_features : array-like of int, default=None
        Indices of the features to treat as Bernoulli. See :class:`MixedNB`.

    gaussian_features : array-like of int, default=None
        Indices of the features to treat as Gaussian. See :class:`MixedNB`.

    var_prior_weight : float, default=1.0
        Pseudo-observations shrinking each Gaussian variance towards the
        pooled variance of the feature (see :class:`MixedNB`).

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

    cpts_ : dict
        Maps every attribute ``i`` with attribute parents to the
        :class:`MixedNB` that models
        :math:`P(x_i \\mid y, \\mathbf{x}_{\\Pi(i)})`. Its classes are the
        observed keys of the augmented class, the mixed-radix index of
        ``(class_index, *parent_values)``.

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
    ):
        self.k = k
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features
        self.var_prior_weight = var_prior_weight

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
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)

        if (
            not isinstance(self.k, numbers.Integral)
            or isinstance(self.k, bool)
            or self.k < 0
        ):
            raise ValueError(f"k must be a non-negative integer, got {self.k!r}")

        # --- 1. Class-level model: prior, parentless conditionals, types ---
        self.backoff_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=self.gaussian_features,
            var_prior_weight=self.var_prior_weight,
        ).fit(X, y)
        types = self.backoff_.feature_types_

        # --- 2. Discretize attributes (structure learning and parents) ---
        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=self.gaussian_features,
            discrete_features=list(self.categorical_features or [])
            + list(self.bernoulli_features or []),
        )
        codes = self._encoder.fit_transform(X)
        cards = self._encoder.cards_

        # --- 3. Structure: order by I(A_i; C), parents by I(A_i; A_j | C) ---
        self.mutual_info_ = _mutual_info_with_class(codes, y_idx, cards, n_classes)
        self.conditional_mutual_info_ = _conditional_mutual_info(
            codes, y_idx, cards, n_classes
        )
        self.feature_order_, self.parents_ = _kdb_structure(
            self.mutual_info_, self.conditional_mutual_info_, self.k
        )

        # --- 4. Parameters: P(x_i | y, x_parents) on the augmented class ---
        X_int = X.astype(np.int64) if types["categorical"] else None
        n_categories = None if X_int is None else X_int.max(axis=0) + 1
        self.cpts_ = {}
        for i, pa in enumerate(self.parents_):
            if len(pa) == 0:
                continue
            y_star = _augmented_keys(y_idx, codes[:, pa].T, n_classes, cards[pa])
            keys, y_star_idx = np.unique(y_star, return_inverse=True)
            cpt = MixedNB(
                alpha=self.alpha,
                categorical_features=[0] if i in types["categorical"] else None,
                bernoulli_features=[0] if i in types["bernoulli"] else None,
                gaussian_features=[0] if i in types["gaussian"] else None,
                var_prior_weight=self.var_prior_weight,
            )
            cpt._fit_encoded(
                X,
                y_star_idx,
                keys,
                X_int=X_int,
                columns=[i],
                n_categories=n_categories,
            )
            self.cpts_[i] = cpt

        # --- 5. Pooled Gaussians for the uninformed estimate ---
        gauss = types["gaussian"]
        Xg = np.asarray(X[:, gauss], dtype=float) if gauss else np.zeros((len(X), 0))
        self._pooled_mean = Xg.mean(axis=0) if gauss else np.zeros(0)
        self._pooled_var = Xg.var(axis=0) + (
            self.backoff_.estimators_["gaussian"].epsilon_ if gauss else 0.0
        )
        return self

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
            out[:, types["categorical"]] = -np.log(self.backoff_._cat_cardinalities)
        if types["bernoulli"]:
            out[:, types["bernoulli"]] = -np.log(2.0)
        return out

    def _joint_log_likelihood(self, X):
        """Computes log P(y, x) for every sample and class.

        Returns
        -------
        jll : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, attributes=["classes_", "parents_", "cpts_"])
        X = validate_data(self, X, reset=False)
        n_classes = len(self.classes_)
        codes, unseen = self._encoder.transform(X)
        cards = self._encoder.cards_

        parentless = [i for i, pa in enumerate(self.parents_) if len(pa) == 0]
        nb_ll = self.backoff_._feature_log_likelihood(X)[:, :, parentless]
        jll = self.backoff_.class_log_prior_ + nb_ll.sum(axis=2)
        uninformed = self._uninformed_log_likelihood(X)
        class_offsets = np.arange(n_classes, dtype=np.int64)[np.newaxis, :]

        for i, cpt in self.cpts_.items():
            pa = self.parents_[i]
            queries = _augmented_keys(
                class_offsets, codes[:, pa].T[:, :, np.newaxis], n_classes, cards[pa]
            )
            pos, hit = _lookup(cpt.classes_, queries)
            hit &= ~unseen[:, pa].any(axis=1, keepdims=True)
            ll = cpt._feature_log_likelihood(X[:, [i]])[:, :, 0]
            jll += np.where(
                hit, np.take_along_axis(ll, pos, axis=1), uninformed[:, [i]]
            )
        return jll

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
        check_is_fitted(self, attributes=["classes_", "parents_", "cpts_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
