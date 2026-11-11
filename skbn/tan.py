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
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval, StrOptions

from ._augmented import (
    _TIE_RTOL,
    _BaseAugmentedNB,
    _conditional_mutual_info,
    _WeightedAugmentedNB,
)


def _maximum_spanning_tree(weights, root):
    """Directed maximum weight spanning tree (Prim's algorithm).

    Ties are broken deterministically: the vertex with the lowest index is
    added first, and a vertex keeps the parent that reached its best weight
    first. Weights within ``_TIE_RTOL`` of the largest one count as tied.

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
    tol = _TIE_RTOL * np.max(np.abs(weights), initial=0.0)
    parents = np.full(n_features, -1, dtype=np.int64)
    in_tree = np.zeros(n_features, dtype=bool)
    in_tree[root] = True
    best = weights[root].astype(np.float64)
    best_parent = np.full(n_features, root, dtype=np.int64)
    for _ in range(n_features - 1):
        frontier = np.where(in_tree, -np.inf, best)
        k = int(np.argmax(frontier >= frontier.max() - tol))
        parents[k] = best_parent[k]
        in_tree[k] = True
        update = ~in_tree & (weights[k] > best + tol)
        best[update] = weights[k][update]
        best_parent[update] = k
    return parents


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

    conditional_mutual_info_ : ndarray of shape (n_features, n_features)
        Empirical conditional mutual information :math:`I(A_i; A_j \\mid C)`
        (in nats) between every pair of (discretized) attributes.

    parents_ : ndarray of shape (n_features,)
        Attribute parent of every attribute in the tree, ``-1`` for the root.

    edges_ : list of tuple of int
        Arcs between attributes, as ``(parent, child)`` pairs of feature
        indices: ``parents_`` in the form :class:`TAN` and :class:`KDB`
        share. :meth:`to_dot` gives the whole network, class included.

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
        self._fit(X, y)
        return self

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


class WeightedTAN(_WeightedAugmentedNB, TAN):
    """
    Discriminatively weighted Tree Augmented Naive Bayes.

    The weighted conditionally Bayesian network (wCCBN) of Zaidi et al. [1]_
    on the structure and the conditionals of :class:`TAN` [2]_: every
    parameter of the network gets an exponent weight,

    .. math::
        P^w(y \\mid \\mathbf{x}) \\propto P(y)^{w_y} \\, P(x_r \\mid y)^{w_{r, x_r, y}}
        \\prod_{i \\neq r} P(x_i \\mid y, x_{\\pi(i)})^{w_{i, x_i, y, x_{\\pi(i)}}},

    and the weights maximise the conditional log-likelihood of the training
    set, penalized by ``l2_reg * ||w - 1||^2``, with L-BFGS-B and analytic
    gradients. The structure and the probabilities are those of :class:`TAN`
    and stay fixed: the weights start at, and are regularized towards, the
    generative model (``w = 1``). The objective is convex in the weights.

    The weights correct what the generative fit gets wrong when the data
    break the independence assumptions of the tree, for instance attributes
    that are correlated or duplicated beyond what one parent explains: they
    scale down the conditionals that count the same evidence twice.

    Weight levels (variables indexing each weight, from coarsest to finest),
    the nested parameterizations of [1]_:

    ===== ===================================
    Level Variables
    ===== ===================================
    1     Factor
    2     Factor x value
    3     Factor x value x class
    4     Factor x value x class x parents
    ===== ===================================

    A factor is the class prior :math:`P(y)` or the conditional of an
    attribute; the class prior has a single value and no parents, so it gets
    one weight per class at levels 3 and 4. Level 4 (the default), one weight
    per parameter of the network, is the wCCBN of [1]_: a reparameterization
    of the discriminative learning of the parameters of that network, which
    [1]_ found faster to optimize. The value of a Gaussian child is not discretized, so its
    weights ignore the "value" dimension. Only the combinations that occur in
    training get a weight; the others share one weight that keeps its
    generative value, so the number of weights is bounded by the data. This is
    :class:`WeightedAnDE` with ``weight_scope="factor"`` and
    ``mean="geometric"`` applied to the single network of TAN. A test instance
    in a parent state never seen in training gets the uninformed conditional
    of :class:`TAN`. The fit keeps the per-factor log-likelihoods of the
    training set in memory, about ``12 * n_samples * n_classes * (n_features +
    1)`` bytes, and each evaluation of the objective costs as much.

    Parameters
    ----------
    root : int, default=0
        Index of the attribute used as root of the tree, in
        ``[0, n_features - 1]``. The root does not change the edges of the
        tree, only their direction, but with smoothing (``alpha > 0``) the
        estimates, and so the predictions, depend on the direction.

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
        and to count and evaluate the conditionals; the weights are optimized
        in a single thread. ``None`` means 1 and ``-1`` all the processors.

    weight_level : int, default=4
        Granularity of the weights, from 1 (coarsest) to 4 (finest); see the
        table above.

    l2_reg : float, default=1.0
        Non-negative L2 regularization towards the generative solution
        (``w = 1``), added to the summed conditional log-likelihood.

    max_iter : int, default=100
        Maximum number of L-BFGS-B iterations (at least 1). A
        :class:`~sklearn.exceptions.ConvergenceWarning` is raised at the end of
        :meth:`fit` when L-BFGS-B stops at ``max_iter`` before converging;
        increase ``max_iter`` or ``tol`` in that case.

    tol : float or None, default=None
        Stopping tolerance of L-BFGS-B on the relative reduction of the
        objective and on the projected gradient; None uses SciPy's defaults.

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

    edges_ : list of tuple of int
        Arcs between attributes, as ``(parent, child)`` pairs of feature
        indices: ``parents_`` in the form :class:`TAN` and :class:`KDB`
        share. :meth:`to_dot` gives the whole network, class included.

    backoff_ : MixedNB
        Naive Bayes on the original class. Provides the class prior, the root
        conditional and the detected feature types (``feature_types_``).

    learned_weights_ : ndarray of shape (n_weights,)
        Learned weights ``w``.

    n_weights_ : int
        Number of weights, including the shared one for unseen combinations.

    prior_weights_ : ndarray of shape (n_classes,) or (1,)
        Weights of the class prior, taken from ``learned_weights_``.

    feature_weights_ : list of ndarray
        Weights of the conditional of every attribute, taken from
        ``learned_weights_``: a row per combination that ``weight_level``
        tells apart and a column per class when it has one. At level 1
        ``feature_weights_[i]`` is the single weight of attribute ``i``.

    n_iter_ : int
        Number of L-BFGS-B iterations.

    See Also
    --------
    TAN : The generative model.
    WeightedKDB : The same weighting on a KDB network.
    WeightedAnDE : Factor weights inside the SPnDEs of an AnDE ensemble.

    References
    ----------
    .. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
           Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
           Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.
    .. [2] Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian network
           classifiers. Machine Learning, 29(2-3), 131-163.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import log_loss
    >>> from skbn import TAN, WeightedTAN
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randint(0, 2, size=300)
    >>> x = rng.normal(size=300) + y
    >>> X = np.column_stack([x, x, x, rng.normal(size=300) + y])  # x three times
    >>> clf = WeightedTAN(weight_level=1).fit(X, y)
    >>> clf.n_weights_  # the class prior, four attributes and the unseen key
    6
    >>> clf.learned_weights_.round(1)  # the copies of x count less
    array([1. , 1.1, 0.3, 0.3, 1. , 1. ])
    >>> tan = TAN().fit(X, y)
    >>> bool(log_loss(y, clf.predict_proba(X)) < log_loss(y, tan.predict_proba(X)))
    True
    """

    _parameter_constraints: dict = {
        **TAN._parameter_constraints,
        **_WeightedAugmentedNB._weight_constraints,
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
        weight_level=4,
        l2_reg=1.0,
        max_iter=100,
        tol=None,
    ):
        super().__init__(
            root=root,
            n_bins=n_bins,
            strategy=strategy,
            alpha=alpha,
            categorical_features=categorical_features,
            bernoulli_features=bernoulli_features,
            gaussian_features=gaussian_features,
            var_prior_weight=var_prior_weight,
            n_jobs=n_jobs,
        )
        self.weight_level = weight_level
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol
