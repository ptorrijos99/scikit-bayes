"""Naive Bayes classifier for heterogeneous (mixed-type) data."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ._core import _auto_type, _feature_indices


def _fit_gaussian_nb(X, y_idx, classes, var_smoothing):
    """GaussianNB with the estimates of ``GaussianNB.fit``, computed without a
    loop over classes (the augmented classes of AnDE can number in the hundreds)."""
    n_classes = len(classes)
    count = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    theta = (
        np.column_stack(
            [
                np.bincount(y_idx, weights=X[:, j], minlength=n_classes)
                for j in range(X.shape[1])
            ]
        )
        / count[:, np.newaxis]
    )
    sq = (X - theta[y_idx]) ** 2
    var = (
        np.column_stack(
            [
                np.bincount(y_idx, weights=sq[:, j], minlength=n_classes)
                for j in range(X.shape[1])
            ]
        )
        / count[:, np.newaxis]
    )
    est = GaussianNB(var_smoothing=var_smoothing)
    est.epsilon_ = var_smoothing * np.var(X, axis=0).max()
    est.theta_ = theta
    est.var_ = var + est.epsilon_
    est.class_count_ = count
    est.class_prior_ = count / count.sum()
    est.classes_ = classes
    est.n_features_in_ = X.shape[1]
    return est


def _fit_categorical_nb(X, y_idx, classes, alpha, n_categories):
    """CategoricalNB with the estimates of ``CategoricalNB.fit``, using a single
    ``bincount`` over all features instead of a loop over features and classes."""
    n_classes = len(classes)
    cards = np.asarray(n_categories, dtype=np.int64)
    # Feature j owns the cells [offsets[j], offsets[j] + n_classes * cards[j])
    offsets = np.concatenate([[0], np.cumsum(n_classes * cards)[:-1]])
    flat = np.bincount(
        (offsets + y_idx[:, np.newaxis] * cards + X).ravel(),
        minlength=int(n_classes * cards.sum()),
    ).astype(np.float64)
    # Smoothed log-probabilities of every cell, normalised per (feature, class) row
    row_len = np.repeat(cards, n_classes)
    row_start = np.concatenate([[0], np.cumsum(row_len)[:-1]])
    smoothed = flat + alpha
    log_cells = np.log(smoothed) - np.repeat(
        np.log(np.add.reduceat(smoothed, row_start)), row_len
    )
    est = CategoricalNB(alpha=alpha, min_categories=n_categories)
    blocks = [slice(o, o + n_classes * r) for o, r in zip(offsets, cards)]
    est.category_count_ = [flat[b].reshape(n_classes, -1) for b in blocks]
    est.feature_log_prob_ = [log_cells[b].reshape(n_classes, -1) for b in blocks]
    est.class_count_ = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    est.class_log_prior_ = np.log(est.class_count_) - np.log(est.class_count_.sum())
    est.n_categories_ = np.asarray(n_categories, dtype=np.int64)
    est.classes_ = classes
    est.n_features_in_ = X.shape[1]
    return est


def _bernoulli_thresholds(X):
    """Threshold that binarises every column of ``X`` (``x > t`` is 1).

    A column with values on both sides of 0 keeps the convention of
    ``BernoulliNB`` (``t = 0``); otherwise ``t`` is the midpoint of its range,
    so that two-valued attributes such as {1, 2} or {0.3, 0.7} keep both
    values apart.
    """
    if X.shape[1] == 0:
        return np.zeros(0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    return np.where((lo <= 0) & (hi > 0), 0.0, (lo + hi) / 2.0)


def _encode_categories(col, categories):
    """Codes of ``col`` in the sorted array ``categories`` and a mask of the
    values that are not in it (their code is 0)."""
    pos = np.minimum(np.searchsorted(categories, col), len(categories) - 1)
    known = categories[pos] == col
    return np.where(known, pos, 0), ~known


def _fit_bernoulli_nb(X_bool, y_idx, classes, alpha):
    """BernoulliNB with the estimates of ``BernoulliNB.fit``, without the dense
    one-hot label matrix that sklearn builds (n_samples x n_classes)."""
    n_classes = len(classes)
    class_count = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    feature_count = np.column_stack(
        [
            np.bincount(y_idx, weights=X_bool[:, j], minlength=n_classes)
            for j in range(X_bool.shape[1])
        ]
    )
    est = BernoulliNB(alpha=alpha)
    est.classes_ = classes
    est.class_count_ = class_count
    est.feature_count_ = feature_count
    est.feature_log_prob_ = np.log(feature_count + alpha) - np.log(
        class_count[:, np.newaxis] + 2 * alpha
    )
    est.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())
    est.n_features_in_ = X_bool.shape[1]
    return est


class MixedNB(ClassifierMixin, BaseEstimator):
    """
    Mixed Naive Bayes classifier for heterogeneous data.

    This classifier is designed to handle datasets with a mix of continuous
    (Gaussian), categorical, and binary (Bernoulli) features. It internally
    uses scikit-learn's GaussianNB, CategoricalNB, and BernoulliNB on the
    respective feature subsets.

    Features are classified and handled as follows:

    1.  User-defined: the constructor parameters fix the type of any feature.
        The three sets of features must be disjoint.
    2.  Auto-detection: the remaining features are typed during `fit`:

        -   Constant features carry no information and are ignored.
        -   Features with exactly two unique values are Bernoulli.
        -   Integer-valued features (integer dtype, or floats without
            fractional part) that are non-negative and have from 3 to
            ``_MAX_AUTO_CATEGORIES`` (20) unique values are categorical.
        -   The rest (fractional or negative values, or integer values with
            more distinct values, such as ages or amounts) are Gaussian.

        The detection only looks at the values, so declaring the types is
        the safe choice whenever they are known.

    Categorical features may hold any numeric codes: they are label-encoded
    on the values seen in training, so a category is a value, not a position.
    A value never seen in training contributes nothing to the prediction, as a
    missing value would (the feature is marginalised for that sample).

    Bernoulli features are binarised as ``x > t``, with ``t = 0`` (the
    convention of `BernoulliNB`) when the training values lie on both sides of
    0, and the midpoint of their range otherwise, so that a two-valued
    attribute such as {1, 2} keeps its two values apart.

    Gaussian features whose training variance is zero are ignored, since
    their density would be degenerate and the same for every class.

    Read more in the :ref:`User Guide <mixed_naive_bayes>`.

    Parameters
    ----------
    categorical_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the features to treat as categorical.

    bernoulli_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the features to treat as Bernoulli.

    gaussian_features : array-like of int or bool, default=None
        Indices (or a boolean mask) of the features to treat as Gaussian.

    var_smoothing : float, default=1e-9
        Non-negative portion of the largest variance of all Gaussian features
        that is added to variances for calculation stability. Passed to
        `GaussianNB`.

    alpha : float, default=1.0
        Non-negative additive (Laplace/Lidstone) smoothing parameter. Passed to
        `CategoricalNB` and `BernoulliNB`.

    var_prior_weight : float, default=0.0
        Non-negative number of pseudo-observations used to shrink each class-conditional
        variance towards the pooled (class-independent) variance of the
        feature: ``var_c = (n_c * var_c + nu * var_pooled) / (n_c + nu)``.
        It plays for Gaussian features the role that ``alpha`` plays for
        discrete ones, preventing degenerate densities in classes with very
        few samples. ``0`` reproduces `GaussianNB` exactly.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    class_log_prior_ : ndarray of shape (n_classes,)
        Log of the empirical frequency of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    feature_types_ : dict
        A dictionary mapping feature type ('gaussian', 'categorical',
        'bernoulli') to the indices of the features of that type. Undeclared
        constant features and Gaussian features with zero variance are in none
        of the lists.

    estimators_ : dict
        A dictionary containing the fitted Naive Bayes estimators for each
        feature type.

    categories_ : list of ndarray
        Sorted training values of each categorical feature, in the order of
        ``feature_types_["categorical"]``.

    bernoulli_thresholds_ : ndarray of shape (n_bernoulli_features,)
        Threshold ``t`` that binarises each Bernoulli feature as ``x > t``.

    See Also
    --------
    GaussianNB : Naive Bayes classifier for Gaussian features.
    CategoricalNB : Naive Bayes classifier for categorical features.
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.

    Examples
    --------
    >>> import numpy as np
    >>> from skbn.mixed_nb import MixedNB
    >>> # Data: [Gaussian, Bernoulli, Categorical (3 cats)]
    >>> X = np.array([
    ...     [0.5, 0, 0],
    ...     [-1.2, 1, 1],
    ...     [0.6, 1, 2],
    ...     [-0.1, 0, 0],
    ...     [2.5, 1, 1],
    ...     [-3.0, 0, 2]
    ... ])
    >>> y = np.array([0, 1, 1, 0, 1, 0])
    >>> clf = MixedNB()
    >>> clf.fit(X, y)
    MixedNB()
    >>> clf.predict([[-0.8, 1, 1]])
    array([1])
    """

    _parameter_constraints: dict = {
        "categorical_features": ["array-like", None],
        "bernoulli_features": ["array-like", None],
        "gaussian_features": ["array-like", None],
        "var_smoothing": [Interval(Real, 0, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "var_prior_weight": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        var_smoothing=1e-9,
        alpha=1.0,
        var_prior_weight=0.0,
    ):
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.var_prior_weight = var_prior_weight

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the Mixed Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y)
        classes = unique_labels(y)
        return self._fit_encoded(X, np.searchsorted(classes, y), classes)

    def _fit_encoded(
        self, X, y_idx, classes, X_int=None, columns=None, categories=None
    ):
        """Fit on a validated ``X`` with labels given as indices into ``classes``.

        TAN and KDB fit one MixedNB per attribute on data they have already
        validated and encoded; this entry point skips the repeated validation
        and label encoding. When ``columns`` is given, the model is fitted on
        ``X[:, columns]`` (feature indices refer to positions in ``columns``)
        without materialising that submatrix. ``categories`` optionally gives,
        for every column of ``X``, the sorted categories of a categorical
        attribute (None for the others), and ``X_int`` the codes of those
        columns in them; those models compute both once for all attributes.
        """
        if columns is None:
            columns = np.arange(X.shape[1])
        columns = np.asarray(columns)

        def cols(local):
            return columns[np.asarray(local, dtype=int)]

        self.classes_ = classes
        self.n_features_in_ = len(columns)

        # --- Parameter and Feature Type Validation ---
        cat_feats = _feature_indices(
            self.categorical_features, self.n_features_in_, "categorical_features"
        )
        bern_feats = _feature_indices(
            self.bernoulli_features, self.n_features_in_, "bernoulli_features"
        )
        gauss_feats = _feature_indices(
            self.gaussian_features, self.n_features_in_, "gaussian_features"
        )

        # Check disjointness
        all_indices = cat_feats + bern_feats + gauss_feats
        if len(set(all_indices)) != len(all_indices):
            raise ValueError(
                "Feature sets (categorical, bernoulli, gaussian) must be disjoint."
            )

        self.feature_types_ = {
            "gaussian": gauss_feats,
            "categorical": cat_feats,
            "bernoulli": bern_feats,
        }

        user_defined_indices = set(cat_feats) | set(bern_feats) | set(gauss_feats)

        # Auto-detect remaining features (constant ones are ignored)
        for i in range(self.n_features_in_):
            if i not in user_defined_indices:
                kind = _auto_type(X[:, columns[i]])
                if kind is not None:
                    self.feature_types_[kind].append(i)

        # A Gaussian feature with no variance has the same degenerate density
        # in every class: it is ignored, as an undeclared constant feature is
        gauss = self.feature_types_["gaussian"]
        if gauss:
            constant = np.ptp(X[:, cols(gauss)], axis=0) == 0
            self.feature_types_["gaussian"] = [
                i for i, c in zip(gauss, constant) if not c
            ]

        # --- Fit Sub-estimators ---
        self.estimators_ = {}
        self.categories_ = []
        self.bernoulli_thresholds_ = np.zeros(0)

        if self.feature_types_["gaussian"]:
            X_gauss = X[:, cols(self.feature_types_["gaussian"])]
            gauss_nb = _fit_gaussian_nb(
                X_gauss, y_idx, self.classes_, self.var_smoothing
            )
            if self.var_prior_weight > 0:
                # Shrink class-conditional variances towards the pooled
                # variance (normal-inverse-gamma style pseudo-observations).
                pooled_var = np.var(X_gauss, axis=0) + gauss_nb.epsilon_
                n_c = gauss_nb.class_count_[:, np.newaxis]
                nu = self.var_prior_weight
                gauss_nb.var_ = (n_c * gauss_nb.var_ + nu * pooled_var) / (n_c + nu)
            self.estimators_["gaussian"] = gauss_nb

        if self.feature_types_["categorical"]:
            g = cols(self.feature_types_["categorical"])
            if categories is None:
                self.categories_ = [np.unique(X[:, j]) for j in g]
                X_cat = np.column_stack(
                    [
                        _encode_categories(X[:, j], c)[0]
                        for j, c in zip(g, self.categories_)
                    ]
                )
            else:
                self.categories_ = [categories[j] for j in g]
                X_cat = X_int[:, g]
            self._cat_cardinalities = np.array(
                [len(c) for c in self.categories_], dtype=int
            )
            self.estimators_["categorical"] = _fit_categorical_nb(
                X_cat, y_idx, self.classes_, self.alpha, self._cat_cardinalities
            )

        if self.feature_types_["bernoulli"]:
            g = cols(self.feature_types_["bernoulli"])
            self.bernoulli_thresholds_ = _bernoulli_thresholds(X[:, g])
            self.estimators_["bernoulli"] = _fit_bernoulli_nb(
                X[:, g] > self.bernoulli_thresholds_, y_idx, self.classes_, self.alpha
            )

        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        return self

    def _categorical_codes(self, X):
        """Codes of the categorical features of a validated ``X`` and a mask of
        the values never seen in training, each of shape
        (n_samples, n_categorical_features)."""
        indices = self.feature_types_["categorical"]
        codes = np.zeros((X.shape[0], len(indices)), dtype=np.int64)
        unseen = np.zeros(codes.shape, dtype=bool)
        for k, (j, cats) in enumerate(zip(indices, self.categories_)):
            codes[:, k], unseen[:, k] = _encode_categories(X[:, j], cats)
        return codes, unseen

    def _shared_encoding(self, X):
        """Categorical encoding of a validated training ``X`` for the models an
        ensemble fits on its columns (see ``_fit_encoded``): the categories of
        every column (None for the non-categorical ones) and an integer array
        with the codes of the categorical columns."""
        categories = [None] * X.shape[1]
        X_int = np.zeros(X.shape, dtype=np.int64)
        indices = self.feature_types_["categorical"]
        if indices:
            X_int[:, indices] = self._categorical_codes(X)[0]
            for j, cats in zip(indices, self.categories_):
                categories[j] = cats
        return X_int, categories

    def _feature_log_likelihood(self, X, class_idx=None):
        """Per-feature class-conditional log-likelihoods.

        ``class_idx`` of shape (n_samples, k) restricts the evaluation of each
        sample to ``k`` classes (indices into ``classes_``); by default every
        class is evaluated.

        Returns
        -------
        ndarray of shape (n_samples, k, n_features)
            Entry ``[i, c, j]`` is ``log P(x_ij | class_idx[i, c])``. Features
            ignored at fit time (constant) and categorical values never seen in
            training contribute zeros. The class prior is not included.
        """
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        X = validate_data(self, X, reset=False)
        n_samples = X.shape[0]
        if class_idx is None:
            class_idx = np.broadcast_to(
                np.arange(len(self.classes_)), (n_samples, len(self.classes_))
            )
        out = np.zeros((n_samples, class_idx.shape[1], self.n_features_in_))
        rows = np.arange(n_samples)[:, np.newaxis]

        if "gaussian" in self.estimators_:
            indices = self.feature_types_["gaussian"]
            est = self.estimators_["gaussian"]
            var = est.var_[class_idx]
            diff = X[:, np.newaxis, indices] - est.theta_[class_idx]
            out[:, :, indices] = -0.5 * (diff**2 / var + np.log(2.0 * np.pi * var))

        if "categorical" in self.estimators_:
            indices = self.feature_types_["categorical"]
            est = self.estimators_["categorical"]
            codes, unseen = self._categorical_codes(X)
            for i, j in enumerate(indices):
                ll = est.feature_log_prob_[i][class_idx, codes[rows, i]]
                out[:, :, j] = np.where(unseen[:, [i]], 0.0, ll)

        if "bernoulli" in self.estimators_:
            indices = self.feature_types_["bernoulli"]
            est = self.estimators_["bernoulli"]
            X_bern = (X[:, indices] > self.bernoulli_thresholds_)[:, np.newaxis, :]
            log_p = est.feature_log_prob_[class_idx]
            out[:, :, indices] = np.where(X_bern, log_p, np.log1p(-np.exp(log_p)))

        return np.nan_to_num(out, nan=-1e10, neginf=-1e10)

    def _joint_log_likelihood(self, X):
        """Calculate the unnormalized posterior log probability of X."""
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        X = validate_data(self, X, reset=False)

        jll = np.zeros((X.shape[0], len(self.classes_)))

        # Gaussian features
        if "gaussian" in self.estimators_:
            indices = self.feature_types_["gaussian"]
            X_gauss = X[:, indices]  # shape: (n_samples, n_gauss)
            est = self.estimators_["gaussian"]
            # Compute log P(x|c) for each class i: sum over features of ((x - mu_i)^2 / var_i)
            diff = (
                X_gauss[:, np.newaxis, :] - est.theta_
            )  # (n_samples, n_classes, n_gauss)
            log_prob = -0.5 * np.sum(
                (diff**2) / est.var_, axis=2
            )  # (n_samples, n_classes)
            log_constant = -0.5 * np.sum(
                np.log(2.0 * np.pi * est.var_), axis=1
            )  # (n_classes,)
            jll += (
                log_prob + log_constant
            )  # broadcasting: (n_samples, n_classes) + (n_classes,)

        # Categorical features; a value never seen in training adds nothing
        if "categorical" in self.estimators_:
            est = self.estimators_["categorical"]
            codes, unseen = self._categorical_codes(X)
            for i in range(codes.shape[1]):
                ll = est.feature_log_prob_[i][:, codes[:, i]].T
                jll += np.where(unseen[:, [i]], 0.0, ll)

        # Bernoulli features
        if "bernoulli" in self.estimators_:
            indices = self.feature_types_["bernoulli"]
            X_bern = (X[:, indices] > self.bernoulli_thresholds_).astype(np.float64)
            est = self.estimators_["bernoulli"]
            log_prob_pos = est.feature_log_prob_
            log_prob_neg = np.log1p(-np.exp(log_prob_pos))  # numerically safer
            jll += X_bern @ (log_prob_pos - log_prob_neg).T
            jll += np.sum(log_prob_neg, axis=1)

        return np.nan_to_num(jll + self.class_log_prior_, nan=-1e10, neginf=-1e10)

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
        log_prob_x = np.logaddexp.reduce(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

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
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        X = validate_data(self, X, reset=False)
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
