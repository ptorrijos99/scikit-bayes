"""Naive Bayes classifier for heterogeneous (mixed-type) data."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from ._core import (
    _auto_types,
    _check_label_types,
    _discrete_codes,
    _encode_columns,
    _feature_indices,
    _label_feature_indices,
    _LabelInputMixin,
    _unique_values,
    _validate_input,
)
from ._kernels import _categorical_counts, _class_moments


def _fit_gaussian_nb(X, y_idx, classes, var_smoothing):
    """GaussianNB with the estimates of ``GaussianNB.fit``, from two compiled
    passes over the samples."""
    count, theta, var = _class_moments(X, y_idx, len(classes))
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
    """CategoricalNB with the estimates of ``CategoricalNB.fit``, counted in a
    single compiled pass over the samples instead of a loop over features and
    classes."""
    n_classes = len(classes)
    cards = np.asarray(n_categories, dtype=np.int64)
    # Feature j owns the cells [offsets[j], offsets[j] + n_classes * cards[j])
    offsets = np.concatenate([[0], np.cumsum(n_classes * cards)[:-1]])
    flat = _categorical_counts(X, y_idx, offsets, cards, int(n_classes * cards.sum()))
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


class MixedNB(_LabelInputMixin, ClassifierMixin, BaseEstimator):
    """
    Mixed Naive Bayes classifier for heterogeneous data.

    This classifier is designed to handle datasets with a mix of continuous
    (Gaussian), categorical, and binary (Bernoulli) features. It internally
    uses scikit-learn's GaussianNB, CategoricalNB, and BernoulliNB on the
    respective feature subsets.

    Features are classified and handled as follows:

    1.  User-defined: the constructor parameters fix the type of any feature.
        The three sets of features must be disjoint.
    2.  Labels: a pandas ``category`` column (whatever its categories) and a
        column of strings (object or string dtype in a DataFrame, or a column
        of a NumPy object or string array) is categorical, unless declared
        Bernoulli; it cannot be declared Gaussian, and it is ignored when
        constant. Its labels are coded by their position among the sorted
        labels seen in training (for a ``category`` column too, whose order
        of categories is not used, and whose categories absent from the
        training rows get no estimate), so the model is the one fitted on
        these codes declared categorical. Such a column must hold labels in
        `predict` too, and must not hold missing values (impute them first).
    3.  Auto-detection: the remaining features are typed during `fit`:

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
        Sorted training values of each categorical feature (its labels for a
        column of labels), in the order of ``feature_types_["categorical"]``.

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
        X, y = _validate_input(self, X, y)
        classes = unique_labels(y)
        labelled = _label_feature_indices(self)
        self._fit_encoded(X, np.searchsorted(classes, y), classes, labelled)
        if labelled:
            # The labels of the columns of labels, rather than their codes
            encoder = self._label_encoder
            self.categories_ = [
                (
                    encoder.labels[encoder.columns.index(j)][values.astype(np.int64)]
                    if j in labelled
                    else values
                )
                for j, values in zip(
                    self.feature_types_["categorical"], self.categories_
                )
            ]
        return self

    def _fit_encoded(self, X, y_idx, classes, label_columns=(), return_codes=False):
        """Fit on a validated ``X`` with labels given as indices into ``classes``.

        :meth:`fit` validates and encodes the labels before calling it; AnDE
        and its variants, which have validated ``X`` and ``y`` themselves, call
        it directly to fit their class-level back-off model without repeating
        that work. With ``return_codes`` it returns ``(self, codes)``, the
        codes of the categorical features in training, shape (n_samples,
        n_categorical), which those models reuse for their own tables.
        ``label_columns`` are the columns that held labels, coded by
        :func:`~skbn._core._validate_input`: categorical unless declared
        otherwise, and ignored when constant, as any undeclared attribute.
        """
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]

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

        _check_label_types(label_columns, gauss_feats)
        user_defined_indices = set(cat_feats) | set(bern_feats) | set(gauss_feats)
        # Right after the declared ones, as if declared after them
        cat_feats = cat_feats + [
            i
            for i in label_columns
            if i not in user_defined_indices and len(_unique_values(X[:, i])) > 1
        ]
        user_defined_indices.update(label_columns)

        self.feature_types_ = {
            "gaussian": gauss_feats,
            "categorical": cat_feats,
            "bernoulli": bern_feats,
        }

        # Auto-detect remaining features (constant ones are ignored)
        undeclared = [
            i for i in range(self.n_features_in_) if i not in user_defined_indices
        ]
        for i, kind in zip(undeclared, _auto_types(X, undeclared)):
            if kind is not None:
                self.feature_types_[kind].append(i)

        # A Gaussian feature with no variance has the same degenerate density
        # in every class: it is ignored, as an undeclared constant feature is
        gauss = self.feature_types_["gaussian"]
        if gauss:
            constant = np.ptp(X[:, gauss], axis=0) == 0
            self.feature_types_["gaussian"] = [
                i for i, c in zip(gauss, constant) if not c
            ]

        # --- Fit Sub-estimators ---
        self.estimators_ = {}
        self.categories_ = self._categories = []
        self.bernoulli_thresholds_ = np.zeros(0)

        if self.feature_types_["gaussian"]:
            X_gauss = X[:, self.feature_types_["gaussian"]]
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
            g = self.feature_types_["categorical"]
            self._categories, X_cat = _discrete_codes(X, g)
            self.categories_ = self._categories
            self._cat_cardinalities = np.array(
                [len(c) for c in self._categories], dtype=int
            )
            self.estimators_["categorical"] = _fit_categorical_nb(
                X_cat, y_idx, self.classes_, self.alpha, self._cat_cardinalities
            )

        if self.feature_types_["bernoulli"]:
            g = self.feature_types_["bernoulli"]
            self.bernoulli_thresholds_ = _bernoulli_thresholds(X[:, g])
            self.estimators_["bernoulli"] = _fit_bernoulli_nb(
                X[:, g] > self.bernoulli_thresholds_, y_idx, self.classes_, self.alpha
            )

        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        if return_codes:
            if not self.feature_types_["categorical"]:
                X_cat = np.zeros((len(X), 0), dtype=np.int64)
            return self, X_cat
        return self

    def _categorical_codes(self, X):
        """Codes of the categorical features of a validated ``X`` and a mask of
        the values never seen in training, each of shape
        (n_samples, n_categorical_features)."""
        return _encode_columns(X, self.feature_types_["categorical"], self._categories)

    def _joint_log_likelihood(self, X):
        """Calculate the unnormalized posterior log probability of X."""
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        X = _validate_input(self, X, reset=False)

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
        # predict_log_proba validates X; validated twice, a DataFrame would
        # reach the second check as an array and warn about feature names
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
