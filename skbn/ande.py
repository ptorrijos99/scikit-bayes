"""
Averaged n-Dependence Estimators (AnDE) and their discriminative weighting.

This module implements ensembles of SuperParent n-Dependence Estimators
(SPnDEs) that support mixed data types (continuous, categorical, binary)
natively through the "Super-Class" strategy:

1.  **AnDE:** the generative AODE/A2DE ensemble [1]_ [2]_, pooled with the
    arithmetic mean (the standard model) or the geometric mean.
2.  **WeightedAnDE:** discriminative weights learned on top of AnDE, either
    one per component of the ensemble or one per factor of each component.

Both build on the generative fit of the ensemble in :mod:`skbn._spnde`.

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

import warnings
from numbers import Integral, Real
from operator import attrgetter
from typing import NamedTuple, Optional

import numpy as np
from scipy.special import logsumexp
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_is_fitted,
    column_or_1d,
)

from ._core import (
    _LEVEL_DIMS,
    _fit_weights,
    _fold_padding,
    _pad_last,
    _row_chunks,
    _validate_input,
    _warn_not_converged,
)
from ._kernels import (
    _weighted_arithmetic_gradient,
    _weighted_arithmetic_scores,
    _weighted_geometric_gradient,
    _weighted_geometric_scores,
)

# _Ensemble, the class of ``ensemble_``, is also imported here so that the
# models pickled while it was defined in this module still load
from ._spnde import _BaseAnDE, _Ensemble  # noqa: F401

# =============================================================================
# 1. Generative model
# =============================================================================


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

    continuous_children : {"gaussian", "discretized"}, default="gaussian"
        How a continuous attribute is modelled as a child. With
        ``"gaussian"`` it keeps its values and has a Gaussian conditional,
        and only its role as a parent uses the bins of ``n_bins`` and
        ``strategy``. With ``"discretized"`` every continuous attribute is
        replaced by its bin, as a categorical attribute, in both roles: the
        model on discretized data of the original works, without a
        discretizer in a pipeline. Values beyond the training range fall in
        the first or the last bin.

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

    cut_points_ : list of ndarray or None
        Cut-points of the continuous attributes that the model bins, and
        None for the other attributes: bin ``b`` of attribute ``i`` holds the
        values from ``cut_points_[i][b - 1]`` (included) to
        ``cut_points_[i][b]``.

    feature_types_ : dict
        Indices of the attributes of each type the model uses, under the keys
        ``"gaussian"``, ``"categorical"`` and ``"bernoulli"`` (those of
        ``backoff_``). An attribute that is constant in training is in none.

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
        "continuous_children": [StrOptions({"gaussian", "discretized"})],
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
        continuous_children="gaussian",
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
        self.continuous_children = continuous_children

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
        X = _validate_input(self, X, reset=False)
        return self._normalize(self._pooled_scores(X, self.mean == "geometric"))


# =============================================================================
# 2. Discriminative weighting
# =============================================================================


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

    max_iter : int, default=1000
        Maximum number of L-BFGS-B iterations (at least 1); with
        ``modular=True`` it applies to the fit of each component. A
        :class:`~sklearn.exceptions.ConvergenceWarning` is raised at the end of
        :meth:`fit` when L-BFGS-B stops at ``max_iter`` before converging;
        increase ``max_iter`` or ``tol`` in that case.

    tol : float or None, default=1e-6
        Stopping tolerance of L-BFGS-B on the relative reduction of the
        objective and on the projected gradient. None uses SciPy's defaults,
        which are stricter on the objective (about 2e-9): they take about
        twice as many iterations for the same predictions.

    modular : bool, default=False
        If True, the weights of each component are fitted independently on
        that component's own conditional log-likelihood, which is convex. With
        arithmetic component weights only the class-specific levels (3 and 4)
        are identifiable; levels 1 and 2 reduce to the generative AnDE, the
        weights stay at 1 and a :class:`UserWarning` is raised.

    continuous_children : {"gaussian", "discretized"}, default="gaussian"
        How a continuous attribute is modelled as a child. With
        ``"gaussian"`` it keeps its values and has a Gaussian conditional,
        and only its role as a parent uses the bins of ``n_bins`` and
        ``strategy``. With ``"discretized"`` every continuous attribute is
        replaced by its bin, as a categorical attribute, in both roles: the
        model on discretized data of the original works, without a
        discretizer in a pipeline. Values beyond the training range fall in
        the first or the last bin.

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

    cut_points_ : list of ndarray or None
        Cut-points of the continuous attributes that the model bins, and
        None for the other attributes: bin ``b`` of attribute ``i`` holds the
        values from ``cut_points_[i][b - 1]`` (included) to
        ``cut_points_[i][b]``.

    feature_types_ : dict
        Indices of the attributes of each type the model uses, under the keys
        ``"gaussian"``, ``"categorical"`` and ``"bernoulli"`` (those of
        ``backoff_``). An attribute that is constant in training is in none.

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
        "continuous_children": [StrOptions({"gaussian", "discretized"})],
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
        max_iter=1000,
        tol=1e-6,
        modular=False,
        continuous_children="gaussian",
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
        self.continuous_children = continuous_children

    # Projected-gradient tolerance (None: that of ``tol``) and number of
    # corrections of L-BFGS-B
    _gtol = None
    _maxcor = 10

    # --- Parameterization ---------------------------------------------------
    # The parameters of the weighting, read through these private names so
    # that a subclass that fixes them (GaussianALR) overrides them with
    # private class attributes rather than public ones that are no parameters
    _mean = property(attrgetter("mean"))
    _average = property(attrgetter("average"))
    _weight_scope = property(attrgetter("weight_scope"))
    _weight_level = property(attrgetter("weight_level"))
    _modular = property(attrgetter("modular"))

    @property
    def _offset(self):
        """Whether weights are mixture coefficients (added in log-space)
        rather than exponents of the log-terms."""
        return self._weight_scope == "component" and self._mean == "arithmetic"

    @property
    def _pools_posteriors(self):
        """Whether the arithmetic mean pools the components' posteriors (the
        geometric mean of posteriors equals that of joints)."""
        return self._average == "posterior" and self._mean == "arithmetic"

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
        if self._weight_scope == "component":
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
        if self._weight_scope == "component":
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
        dims = _LEVEL_DIMS[self._weight_scope][self._weight_level]
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
        per_factor = self._weight_scope == "factor"
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
        if self._mean == "geometric":
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
        if self._mean == "geometric":
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
            if self._weight_scope == "component" and self._mean == "geometric"
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
        X = _validate_input(self, X, reset=False)
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
        if not self._modular:
            theta = self._optimize(theta, design, Y)
        elif (
            self._offset and "class" not in _LEVEL_DIMS["component"][self._weight_level]
        ):
            warnings.warn(
                (
                    "Modular WeightedAnDE with arithmetic component weights at"
                    f" weight_level {self._weight_level} is equivalent to the"
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
        X = _validate_input(self, X, reset=False)
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
