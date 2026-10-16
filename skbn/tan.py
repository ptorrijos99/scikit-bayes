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

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from .ande import _lookup
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
    n_samples, n_features = codes.shape
    n_c = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    cmi = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            ci, cj = int(cards[i]), int(cards[j])
            key = (y_idx.astype(np.int64) * ci + codes[:, i]) * cj + codes[:, j]
            n_cij = (
                np.bincount(key, minlength=n_classes * ci * cj)
                .reshape(n_classes, ci, cj)
                .astype(np.float64)
            )
            n_ci = n_cij.sum(axis=2, keepdims=True)
            n_cj = n_cij.sum(axis=1, keepdims=True)
            # sum_{c,a,b} P(a,b,c) log[ P(a,b|c) / (P(a|c) P(b|c)) ]
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = n_cij * np.log(n_cij * n_c[:, None, None] / (n_ci * n_cj))
            value = np.sum(terms[n_cij > 0]) / n_samples
            cmi[i, j] = cmi[j, i] = max(value, 0.0)
    return cmi


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


class TAN(ClassifierMixin, BaseEstimator):
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
    estimated by a :class:`MixedNB` fitted on the augmented class
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
        Index of the attribute used as root of the tree. The choice does not
        change the learned (undirected) tree, only the direction of its edges.

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

    conditional_mutual_info_ : ndarray of shape (n_features, n_features)
        Empirical conditional mutual information :math:`I(A_i; A_j \\mid C)`
        (in nats) between every pair of (discretized) attributes.

    parents_ : ndarray of shape (n_features,)
        Attribute parent of every attribute in the tree, ``-1`` for the root.

    backoff_ : MixedNB
        Naive Bayes on the original class. Provides the class prior, the root
        conditional and the detected feature types (``feature_types_``).

    cpts_ : dict
        Maps every non-root attribute ``i`` to the :class:`MixedNB` that models
        :math:`P(x_i \\mid y, x_{\\pi(i)})`. Its classes are the observed keys
        ``class_index * n_parent_values + parent_value``.

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
    >>> clf.parents_
    array([-1,  0,  0])
    >>> clf.score(X, y)
    1.0
    """

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
    ):
        self.root = root
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features
        self.var_prior_weight = var_prior_weight

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
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)

        if (
            not isinstance(self.root, numbers.Integral)
            or isinstance(self.root, bool)
            or not 0 <= self.root < self.n_features_in_
        ):
            raise ValueError(
                f"root must be an integer in [0, {self.n_features_in_ - 1}], got"
                f" {self.root!r}"
            )

        # --- 1. Class-level model: prior, root conditional and feature types ---
        self.backoff_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=self.gaussian_features,
            var_prior_weight=self.var_prior_weight,
        ).fit(X, y)
        types = self.backoff_.feature_types_

        # --- 2. Discretize attributes (structure learning and parents) ---
        self._fit_discretizers(X)

        # --- 3. Structure: Chow-Liu tree on I(A_i; A_j | C) ---
        self.conditional_mutual_info_ = _conditional_mutual_info(
            self._parent_data, y_idx, self._parent_cards, n_classes
        )
        self.parents_ = _maximum_spanning_tree(self.conditional_mutual_info_, self.root)

        # --- 4. Parameters: P(x_i | y, x_parent) on the augmented class ---
        X_int = X.astype(np.int64) if types["categorical"] else None
        n_categories = None if X_int is None else X_int.max(axis=0) + 1
        self.cpts_ = {}
        for i, p in enumerate(self.parents_):
            if p < 0:
                continue
            y_star = y_idx.astype(np.int64) * self._parent_cards[p] + (
                self._parent_data[:, p]
            )
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

    def _fit_discretizers(self, X):
        """Fits one discretizer per attribute and stores the integer codes."""
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

        # Declared feature types take precedence over the dtype heuristic
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
            else:
                est = LabelEncoder()
                self._parent_data[:, i] = est.fit_transform(col)
            self._discretizers_list[i] = est

        self._parent_cards = np.max(self._parent_data, axis=0).astype(int) + 1

    def _discretize(self, X):
        """Maps (validated) X to the integer codes used at fit time.

        Returns the codes (clipped to the training range) and a boolean mask
        of categorical values never seen in training.
        """
        codes = np.zeros(X.shape, dtype=int)
        unseen = np.zeros(X.shape, dtype=bool)
        for i, est in self._discretizers_list.items():
            if isinstance(est, LabelEncoder):
                col = X[:, i]
                known = np.isin(col, est.classes_)
                unseen[:, i] = ~known
                codes[:, i] = est.transform(np.where(known, col, est.classes_[0]))
            else:
                codes[:, i] = est.transform(X[:, [i]]).flatten().astype(int)
        np.clip(codes, 0, self._parent_cards - 1, out=codes)
        return codes, unseen

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
        codes, unseen = self._discretize(X)

        root_ll = self.backoff_._feature_log_likelihood(X)[:, :, self.root]
        jll = self.backoff_.class_log_prior_ + root_ll
        uninformed = self._uninformed_log_likelihood(X)
        class_offsets = np.arange(n_classes, dtype=np.int64)

        for i, cpt in self.cpts_.items():
            p = self.parents_[i]
            queries = (
                class_offsets[np.newaxis, :] * self._parent_cards[p]
                + codes[:, p][:, np.newaxis]
            )
            pos, hit = _lookup(cpt.classes_, queries)
            hit &= ~unseen[:, [p]]
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
