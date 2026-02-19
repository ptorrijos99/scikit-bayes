import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class DecisionTreeDiscretizer(TransformerMixin, BaseEstimator):
    """Supervised discretizer that uses decision trees to find optimal bin boundaries.

    A decision tree is fitted independently for each feature using the target
    variable ``y``.  The internal split thresholds of each tree are used as
    cut-points, providing a supervised alternative to equal-width or
    equal-frequency binning and serving as a practical proxy for MDL-based
    discretization (Fayyad & Irani, 1993).

    When a tree finds no beneficial split for a feature, the behaviour is
    controlled by ``fallback``:

    * ``"single"`` — assign all samples to bin 0 (the supervised criterion
      found no signal; downstream models see a constant column).
    * ``"quantile"`` — fall back to unsupervised equal-frequency binning via
      :class:`~sklearn.preprocessing.KBinsDiscretizer`.  Useful when the
      feature may carry signal through interactions not visible marginally.
    * ``"uniform"`` — fall back to unsupervised equal-width binning via
      :class:`~sklearn.preprocessing.KBinsDiscretizer`.

    .. warning::
        ``"quantile"`` and ``"uniform"`` mix supervised and unsupervised
        criteria within the same transformer.  This is intentional but should
        be noted when interpreting feature importances or model explanations.

    Missing values (``NaN``) are imputed with the per-feature median **only**
    during tree fitting; the same imputer is reused at transform time so that
    the binning of missing values is deterministic and consistent.

    Parameters
    ----------
    n_bins : int, default=10
        Maximum number of bins (leaves) produced per feature.  Must be >= 2.
    criterion : {"gini", "entropy"}, default="entropy"
        Impurity criterion used to evaluate candidate splits.  ``"entropy"``
        is the closest analogue to the MDL criterion.
    min_samples_leaf : int, default=5
        Minimum number of samples required in a leaf node.  Increasing this
        value prevents very fine-grained splits on small datasets.
    fallback : {"single", "quantile", "uniform"}, default="single"
        Strategy for features where the decision tree finds no split.

        ``"single"``
            All samples are assigned to bin 0.  The supervised criterion
            found no marginal signal; the column becomes constant.
        ``"quantile"``
            Fall back to equal-frequency binning using ``n_bins`` bins.
        ``"uniform"``
            Fall back to equal-width binning using ``n_bins`` bins.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the decision tree estimator.  Pass an int
        for reproducible results across multiple function calls.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    thresholds_ : list of ndarray of shape (n_thresholds_i,)
        Sorted split thresholds for each feature ``i``.  An empty array
        indicates that the tree found no beneficial split; the ``fallback``
        strategy is applied for those features.
    imputers_ : list of SimpleImputer or None
        Per-feature imputers fitted on NaN-containing features.  ``None`` for
        features without missing values.
    fallback_discretizers_ : list of KBinsDiscretizer or None
        Fitted fallback discretizers for features with no tree split.
        ``None`` for features that were successfully split by the tree, or
        when ``fallback="single"``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> disc = DecisionTreeDiscretizer(n_bins=5, random_state=0)
    >>> X_disc = disc.fit_transform(X, y)
    >>> X_disc.shape
    (150, 4)
    >>> disc_qnt = DecisionTreeDiscretizer(n_bins=5, fallback="quantile", random_state=0)
    >>> X_disc_qnt = disc_qnt.fit_transform(X, y)
    """

    _VALID_FALLBACKS = {"single", "quantile", "uniform"}

    def __init__(
        self,
        n_bins: int = 10,
        criterion: str = "entropy",
        min_samples_leaf: int = 5,
        fallback: str = "single",
        random_state=None,
    ) -> None:
        self.n_bins = n_bins
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.fallback = fallback
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit one decision tree per feature to determine bin boundaries.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.  May contain ``NaN`` values.
        y : array-like of shape (n_samples,)
            Target class labels used to guide supervised splitting.

        Returns
        -------
        self : DecisionTreeDiscretizer
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``n_bins`` is less than 2 or ``fallback`` is not valid.
        """
        self._validate_params()

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.thresholds_ = []
        self.imputers_ = []
        self.fallback_discretizers_ = []

        for i in range(self.n_features_in_):
            xi = X[:, i].reshape(-1, 1)
            imputer = self._fit_imputer(xi)
            xi_clean = imputer.transform(xi) if imputer is not None else xi

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_leaf_nodes=self.n_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree.fit(xi_clean, y)

            raw = tree.tree_.threshold
            thresholds = np.sort(np.unique(raw[raw != -2]))

            self.thresholds_.append(thresholds)
            self.imputers_.append(imputer)
            self.fallback_discretizers_.append(self._fit_fallback(xi_clean, thresholds))

        return self

    def transform(self, X):
        """Discretize ``X`` using the bin boundaries found during :meth:`fit`.

        Each continuous value is replaced by the integer index of the bin it
        falls into (0-based).  ``NaN`` values are first imputed using the
        per-feature median computed at fit time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to discretize.  May contain ``NaN`` values.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Discretized array with integer bin indices.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If the number of features in ``X`` differs from that seen in
            :meth:`fit`.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            # Standard scikit-learn error message for n_features mismatch
            raise ValueError(
                f"X has {X.shape[1]} features, but DecisionTreeDiscretizer is "
                f"expecting {self.n_features_in_} features as input."
            )

        X_out = np.empty_like(X, dtype=X.dtype)

        for i in range(self.n_features_in_):
            xi = X[:, i].copy()
            imputer = self.imputers_[i]

            if imputer is not None:
                xi = imputer.transform(xi.reshape(-1, 1)).ravel()

            thresholds = self.thresholds_[i]

            if thresholds.size > 0:
                X_out[:, i] = np.digitize(xi, thresholds)
            else:
                fb = self.fallback_discretizers_[i]
                if fb is not None:
                    X_out[:, i] = fb.transform(xi.reshape(-1, 1)).ravel()
                else:
                    X_out[:, i] = 0

        return X_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        """Validate constructor parameters before fitting.

        Raises
        ------
        ValueError
            If any parameter value is outside the accepted range or set.
        """
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins!r}.")
        if self.fallback not in self._VALID_FALLBACKS:
            raise ValueError(
                f"fallback must be one of {self._VALID_FALLBACKS!r}, "
                f"got {self.fallback!r}."
            )

    def _fit_fallback(self, xi: np.ndarray, thresholds: np.ndarray):
        """Return a fitted fallback discretizer if needed, or ``None``.

        A fallback is only created when ``thresholds`` is empty (i.e. the
        decision tree found no split) and ``self.fallback`` is not
        ``"single"``.

        Parameters
        ----------
        xi : ndarray of shape (n_samples, 1)
            Imputed single-feature column.
        thresholds : ndarray of shape (n_thresholds,)
            Split thresholds extracted from the fitted tree.

        Returns
        -------
        discretizer : KBinsDiscretizer or None
        """
        if thresholds.size > 0 or self.fallback == "single":
            return None

        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.fallback,  # "quantile" or "uniform"
        )
        discretizer.fit(xi)
        return discretizer

    @staticmethod
    def _fit_imputer(xi: np.ndarray):
        """Return a fitted median imputer for *xi*, or ``None`` if not needed.

        Parameters
        ----------
        xi : ndarray of shape (n_samples, 1)
            Single-feature column, potentially containing ``NaN``.

        Returns
        -------
        imputer : SimpleImputer or None
        """
        if not np.isnan(xi).any():
            return None

        imputer = SimpleImputer(strategy="median")
        imputer.fit(xi)
        return imputer
