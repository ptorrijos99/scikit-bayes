"""Naive Bayes classifier for heterogeneous (mixed-type) data."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.utils import Bunch
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from ._core import (
    _LEVEL_DIMS,
    _auto_types,
    _check_label_types,
    _discrete_codes,
    _encode_columns,
    _feature_indices,
    _feature_name,
    _fit_weights,
    _fold_padding,
    _format_dot,
    _label_feature_indices,
    _LabelInputMixin,
    _pad_last,
    _resolve_feature,
    _row_chunks,
    _table_frame,
    _unique_values,
    _validate_input,
    _validate_sample_weight,
    _warn_not_converged,
    _weighted_moments,
)
from ._kernels import (
    _categorical_counts,
    _class_moments,
    _weighted_geometric_gradient,
    _weighted_geometric_scores,
)


def _fit_gaussian_nb(X, y_idx, classes, var_smoothing, sw=None):
    """GaussianNB with the estimates of ``GaussianNB.fit``, from two compiled
    passes over the samples, supporting sample weights and missing values."""
    sw_arr = np.empty(0, dtype=np.float64) if sw is None else sw
    count_cj, theta, var = _class_moments(X, y_idx, len(classes), sw_arr)
    pooled_mean, pooled_var = _weighted_moments(X, sw)
    missing_cell = count_cj == 0.0
    if missing_cell.any():
        theta = np.where(missing_cell, pooled_mean[None, :], theta)
        var = np.where(missing_cell, pooled_var[None, :], var)
    est = GaussianNB(var_smoothing=var_smoothing)
    est.epsilon_ = var_smoothing * pooled_var.max()
    est.theta_ = theta
    est.var_ = var + est.epsilon_
    class_count = np.bincount(y_idx, weights=sw, minlength=len(classes)).astype(
        np.float64
    )
    est.class_count_ = class_count
    est.class_prior_ = class_count / class_count.sum()
    est.classes_ = classes
    est.n_features_in_ = X.shape[1]
    return est, count_cj, pooled_var


def _fit_categorical_nb(X, y_idx, classes, alpha, n_categories, sw=None):
    """CategoricalNB with the estimates of ``CategoricalNB.fit``, counted in a
    single compiled pass over the samples, supporting sample weights and
    missing values (negative codes)."""
    n_classes = len(classes)
    cards = np.asarray(n_categories, dtype=np.int64)
    # Feature j owns the cells [offsets[j], offsets[j] + n_classes * cards[j])
    offsets = np.concatenate([[0], np.cumsum(n_classes * cards)[:-1]])
    sw_arr = np.empty(0, dtype=np.float64) if sw is None else sw
    flat = _categorical_counts(
        X, y_idx, offsets, cards, int(n_classes * cards.sum()), sw_arr
    )
    # Smoothed log-probabilities of every cell, normalised per (feature, class) row
    row_len = np.repeat(cards, n_classes)
    row_start = np.concatenate([[0], np.cumsum(row_len)[:-1]])
    smoothed = flat + alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        log_cells = np.log(smoothed) - np.repeat(
            np.log(np.add.reduceat(smoothed, row_start)), row_len
        )
    est = CategoricalNB(alpha=alpha, min_categories=n_categories)
    blocks = [slice(o, o + n_classes * r) for o, r in zip(offsets, cards)]
    est.category_count_ = [flat[b].reshape(n_classes, -1) for b in blocks]
    est.feature_log_prob_ = [log_cells[b].reshape(n_classes, -1) for b in blocks]
    est.class_count_ = np.bincount(y_idx, weights=sw, minlength=n_classes).astype(
        np.float64
    )
    with np.errstate(divide="ignore"):
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
    values apart. Missing values (``NaN``) are ignored.
    """
    if X.shape[1] == 0:
        return np.zeros(0)
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    return np.where((lo <= 0) & (hi > 0), 0.0, (lo + hi) / 2.0)


def _fit_bernoulli_nb(X_col, thresholds, y_idx, classes, alpha, sw=None):
    """BernoulliNB with the estimates of ``BernoulliNB.fit``, supporting
    sample weights and missing values (``NaN``)."""
    n_classes = len(classes)
    n_feats = X_col.shape[1]
    class_count = np.bincount(y_idx, weights=sw, minlength=n_classes).astype(np.float64)
    feature_count = np.zeros((n_classes, n_feats))
    valid_count = np.zeros((n_classes, n_feats))
    is_float = np.issubdtype(X_col.dtype, np.floating)
    for j in range(n_feats):
        col = X_col[:, j]
        if is_float and np.isnan(col).any():
            valid = ~np.isnan(col)
            y_v = y_idx[valid]
            pos = (col[valid] > thresholds[j]).astype(np.float64)
            w_v = np.ones(len(y_v)) if sw is None else sw[valid]
            feature_count[:, j] = np.bincount(
                y_v, weights=pos * w_v, minlength=n_classes
            )
            valid_count[:, j] = np.bincount(y_v, weights=w_v, minlength=n_classes)
        else:
            pos = (col > thresholds[j]).astype(np.float64)
            w_pos = pos if sw is None else pos * sw
            feature_count[:, j] = np.bincount(y_idx, weights=w_pos, minlength=n_classes)
            valid_count[:, j] = class_count
    est = BernoulliNB(alpha=alpha)
    est.classes_ = classes
    est.class_count_ = class_count
    est.feature_count_ = feature_count
    with np.errstate(divide="ignore", invalid="ignore"):
        est.feature_log_prob_ = np.log(feature_count + alpha) - np.log(
            valid_count + 2 * alpha
        )
        est.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())
    est.n_features_in_ = n_feats
    return est, valid_count


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
        `predict` too; missing values (``None``, ``NaN``, ``pd.NA``) are
        supported natively in both `fit` and `predict`.
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
    >>> from skbn import MixedNB
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
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Mixed Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features. May contain missing values
            (``NaN`` or missing labels).

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Non-negative weights applied to individual samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = _validate_input(self, X, y)
        sw = _validate_sample_weight(sample_weight, X)
        classes = unique_labels(y)
        y_idx = np.searchsorted(classes, y)
        if sw is not None and np.any(sw == 0):
            pos = sw > 0
            X, y_idx, sw = X[pos], y_idx[pos], sw[pos]
        labelled = _label_feature_indices(self)
        self._fit_encoded(X, y_idx, classes, labelled, sample_weight=sw)
        self._apply_label_categories(labelled)
        return self

    def _apply_label_categories(self, labelled):
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

    def _fit_encoded(
        self,
        X,
        y_idx,
        classes,
        label_columns=(),
        return_codes=False,
        sample_weight=None,
    ):
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

        # Auto-detect remaining features (constant or all-missing ones are ignored)
        undeclared = [
            i for i in range(self.n_features_in_) if i not in user_defined_indices
        ]
        for i, kind in zip(undeclared, _auto_types(X, undeclared)):
            if kind is not None:
                self.feature_types_[kind].append(i)

        # A Gaussian feature with no variance (or all missing) has a degenerate
        # density in every class: it is ignored, as an undeclared constant is
        gauss = self.feature_types_["gaussian"]
        if gauss:
            _, gvar = _weighted_moments(
                np.asarray(X[:, gauss], dtype=np.float64), sample_weight
            )
            self.feature_types_["gaussian"] = [
                i for i, v in zip(gauss, gvar) if v > 0.0
            ]

        # Declared categorical or Bernoulli features that are all-NaN are ignored
        if np.issubdtype(X.dtype, np.floating):
            for key in ("categorical", "bernoulli"):
                cols = self.feature_types_[key]
                if cols:
                    self.feature_types_[key] = [
                        i for i in cols if not np.all(np.isnan(X[:, i]))
                    ]

        # --- Fit Sub-estimators ---
        self.estimators_ = {}
        self.categories_ = self._categories = []
        self.bernoulli_thresholds_ = np.zeros(0)
        self._gauss_counts_ = None
        self._bern_counts_ = None

        if self.feature_types_["gaussian"]:
            X_gauss = np.asarray(
                X[:, self.feature_types_["gaussian"]], dtype=np.float64
            )
            gauss_nb, count_cj, pooled_var = _fit_gaussian_nb(
                X_gauss, y_idx, self.classes_, self.var_smoothing, sw=sample_weight
            )
            self._gauss_counts_ = count_cj
            if self.var_prior_weight > 0:
                # Shrink class-conditional variances towards the pooled
                # variance (normal-inverse-gamma style pseudo-observations).
                pooled = pooled_var + gauss_nb.epsilon_
                nu = self.var_prior_weight
                gauss_nb.var_ = (count_cj * gauss_nb.var_ + nu * pooled) / (
                    count_cj + nu
                )
            self.estimators_["gaussian"] = gauss_nb

        if self.feature_types_["categorical"]:
            g = self.feature_types_["categorical"]
            self._categories, X_cat = _discrete_codes(X, g)
            self.categories_ = self._categories
            self._cat_cardinalities = np.array(
                [len(c) for c in self._categories], dtype=int
            )
            self.estimators_["categorical"] = _fit_categorical_nb(
                X_cat,
                y_idx,
                self.classes_,
                self.alpha,
                self._cat_cardinalities,
                sw=sample_weight,
            )

        if self.feature_types_["bernoulli"]:
            g = self.feature_types_["bernoulli"]
            X_bern = X[:, g]
            self.bernoulli_thresholds_ = _bernoulli_thresholds(X_bern)
            bern_nb, valid_count = _fit_bernoulli_nb(
                X_bern,
                self.bernoulli_thresholds_,
                y_idx,
                self.classes_,
                self.alpha,
                sw=sample_weight,
            )
            self.estimators_["bernoulli"] = bern_nb
            self._bern_counts_ = valid_count

        class_counts = np.bincount(
            y_idx, weights=sample_weight, minlength=len(self.classes_)
        ).astype(np.float64)
        self.class_count_ = class_counts
        with np.errstate(divide="ignore"):
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

        # Gaussian features; missing values (NaN) are marginalised out
        if "gaussian" in self.estimators_:
            indices = self.feature_types_["gaussian"]
            X_gauss = np.asarray(X[:, indices], dtype=np.float64)
            est = self.estimators_["gaussian"]
            diff = X_gauss[:, np.newaxis, :] - est.theta_
            ll = -0.5 * ((diff**2) / est.var_ + np.log(2.0 * np.pi * est.var_))
            if np.isnan(X_gauss).any():
                ll = np.where(np.isnan(X_gauss[:, np.newaxis, :]), 0.0, ll)
            jll += np.sum(ll, axis=2)

        # Categorical features; missing or unseen values add nothing
        if "categorical" in self.estimators_:
            est = self.estimators_["categorical"]
            codes, unseen = self._categorical_codes(X)
            for i in range(codes.shape[1]):
                ll = est.feature_log_prob_[i][:, codes[:, i]].T
                jll += np.where(unseen[:, [i]], 0.0, ll)

        # Bernoulli features; missing values (NaN) are marginalised out
        if "bernoulli" in self.estimators_:
            indices = self.feature_types_["bernoulli"]
            X_b = X[:, indices]
            est = self.estimators_["bernoulli"]
            log_prob_pos = est.feature_log_prob_
            log_prob_neg = np.log1p(-np.exp(log_prob_pos))  # numerically safer
            if np.issubdtype(X_b.dtype, np.floating) and np.isnan(X_b).any():
                missing = np.isnan(X_b)
                bits = (X_b > self.bernoulli_thresholds_).astype(np.float64)
                for i in range(len(indices)):
                    ll = np.where(
                        bits[:, [i]] == 1.0,
                        log_prob_pos[:, i][None, :],
                        log_prob_neg[:, i][None, :],
                    )
                    jll += np.where(missing[:, [i]], 0.0, ll)
            else:
                X_bern = (X_b > self.bernoulli_thresholds_).astype(np.float64)
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

    @property
    def edges_(self):
        """Arcs between attributes: always empty for Naive Bayes."""
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        return []

    def to_dot(self, feature_names=None, class_name="class"):
        """Return the Naive Bayes network in Graphviz DOT format.

        Parameters
        ----------
        feature_names : sequence of str, default=None
            Names of the attributes. Defaults to ``feature_names_in_`` when
            fitted on a DataFrame, otherwise ``x0``, ``x1``, ...

        class_name : str, default="class"
            Label of the class node.

        Returns
        -------
        dot : str
            A ``digraph`` with an arc from the class to every active attribute.
        """
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        used = {j for cols in self.feature_types_.values() for j in cols}
        return _format_dot(
            self, used, [], feature_names=feature_names, class_name=class_name
        )

    def _active_feature(self, feature):
        """Resolve ``feature`` (by index or name) and check that it is active."""
        check_is_fitted(self, attributes=["classes_", "estimators_"])
        idx = _resolve_feature(self, feature)
        for kind in ("gaussian", "categorical", "bernoulli"):
            if idx in self.feature_types_[kind]:
                return idx, kind, self.feature_types_[kind].index(idx)
        raise ValueError(
            f"Feature {idx} is constant in training: the model ignores it"
            " and it has no conditional."
        )

    def conditional_table(self, feature, as_frame=False):
        """Return the class-conditional distribution of an attribute.

        Parameters
        ----------
        feature : int or str
            The attribute, by its index or by its name in
            ``feature_names_in_``.

        as_frame : bool, default=False
            If True, return a pandas DataFrame indexed by the class, with a
            column per value of the attribute (or ``mean`` and ``var`` for a
            Gaussian attribute).

        Returns
        -------
        table : :class:`~sklearn.utils.Bunch` or DataFrame
            With ``as_frame=False``, a dictionary-like object with the same
            fields as :meth:`TAN.conditional_table` and
            :meth:`KDB.conditional_table`: ``feature``, ``parents`` (empty),
            ``classes``, ``parent_values`` (empty), ``count``,
            ``feature_values``, ``proba``, ``mean`` and ``var``.
        """
        idx, kind, pos = self._active_feature(feature)
        n_classes = len(self.classes_)
        parent_values = np.empty((n_classes, 0), dtype=object)
        count = self.class_count_.copy()
        feature_values = proba = mean = var = None
        if kind == "gaussian":
            est = self.estimators_["gaussian"]
            mean = est.theta_[:, pos].copy()
            var = est.var_[:, pos].copy()
            if hasattr(self, "_gauss_counts_"):
                count = self._gauss_counts_[:, pos].copy()
        elif kind == "categorical":
            est = self.estimators_["categorical"]
            feature_values = self.categories_[pos]
            proba = np.exp(est.feature_log_prob_[pos])
            count = est.category_count_[pos].sum(axis=1)
        else:
            est = self.estimators_["bernoulli"]
            feature_values = np.array([False, True])
            p_pos = np.exp(est.feature_log_prob_[:, pos])
            proba = np.column_stack([1.0 - p_pos, p_pos])
            if hasattr(self, "_bern_counts_"):
                count = self._bern_counts_[:, pos].copy()
        table = Bunch(
            feature=idx,
            parents=[],
            classes=self.classes_.copy(),
            parent_values=parent_values,
            count=count,
            feature_values=feature_values,
            proba=proba,
            mean=mean,
            var=var,
        )
        if not as_frame:
            return table
        if proba is None:
            return _table_frame({"mean": mean, "var": var}, [table.classes], ["class"])
        frame = _table_frame(proba, [table.classes], ["class"], feature_values)
        return frame.rename_axis(columns=_feature_name(self, idx))


class WeightedNB(MixedNB):
    """Weighted Mixed Naive Bayes classifier (WANBIA / wCCBN on Naive Bayes).

    Extends :class:`MixedNB` with discriminative exponent weights on the class
    prior and every attribute conditional, learned by maximising the
    conditional log-likelihood with L-BFGS-B:

    .. math::
        \\log P^w(y \\mid \\mathbf{x}) = w_y \\log P(y)
        + \\sum_{i} w_{i, x_i, y} \\log P(x_i \\mid y) - \\log Z(\\mathbf{x}).

    The objective is convex. The weights start at, and are regularized
    towards (``l2_reg``), the generative model ``w = 1``, which predicts as
    :class:`MixedNB`. ``weight_level=1`` gives one weight per factor
    (WANBIA-C [1]_), ``weight_level=2`` one weight per factor and discrete
    value, and ``weight_level=3`` (default) one weight per factor, discrete
    value and class (the weighted CCBN of Zaidi et al. [2]_ on the Naive
    Bayes structure).

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
        added to variances for calculation stability.

    alpha : float, default=1.0
        Non-negative additive (Laplace/Lidstone) smoothing parameter.

    var_prior_weight : float, default=0.0
        Non-negative number of pseudo-observations shrinking each Gaussian
        variance towards the pooled variance of the feature.

    weight_level : int, default=3
        Granularity of the weights, from 1 (one weight per factor: WANBIA-C)
        to 3 (one weight per factor, discrete value and class).

    l2_reg : float, default=1.0
        Non-negative L2 regularization towards the generative solution
        (``w = 1``), added to the summed conditional log-likelihood.

    max_iter : int, default=1000
        Maximum number of L-BFGS-B iterations (at least 1).

    tol : float or None, default=1e-6
        Stopping tolerance of L-BFGS-B on the relative reduction of the
        objective and on the projected gradient.

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
        Indices of the attributes of each type.

    estimators_ : dict
        Fitted Naive Bayes sub-estimators of the generative phase.

    categories_ : list of ndarray
        Sorted training values of each categorical feature.

    bernoulli_thresholds_ : ndarray of shape (n_bernoulli_features,)
        Binarisation threshold of each Bernoulli feature.

    learned_weights_ : ndarray of shape (n_weights,)
        Learned weights ``w``.

    n_weights_ : int
        Number of weights, including the shared one for unseen combinations.

    prior_weights_ : ndarray of shape (n_classes,) or (1,)
        Weights of the class prior, taken from ``learned_weights_``.

    feature_weights_ : list of ndarray
        Weights of the conditional of every attribute, taken from
        ``learned_weights_``.

    n_iter_ : int
        Number of L-BFGS-B iterations.

    See Also
    --------
    MixedNB : The generative model.
    WeightedTAN : Discriminative weights on a TAN network.
    WeightedKDB : Discriminative weights on a KDB network.

    References
    ----------
    .. [1] Zaidi, N. A., Cerquides, J., Carman, M. J., & Webb, G. I. (2013).
           Alleviating naive Bayes attribute independence assumption by
           attribute weighting. Journal of Machine Learning Research, 14,
           1947-1988.
    .. [2] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
           Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
           Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import log_loss
    >>> from skbn import MixedNB, WeightedNB
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randint(0, 2, size=300)
    >>> x = rng.normal(size=300) + y
    >>> X = np.column_stack([x, x, x, rng.normal(size=300) + y])
    >>> clf = WeightedNB(weight_level=1).fit(X, y)
    >>> nb = MixedNB().fit(X, y)
    >>> bool(log_loss(y, clf.predict_proba(X)) < log_loss(y, nb.predict_proba(X)))
    True
    """

    _parameter_constraints: dict = {
        **MixedNB._parameter_constraints,
        "weight_level": [Interval(Integral, 1, 3, closed="both")],
        "l2_reg": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left"), None],
    }

    def __init__(
        self,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        var_smoothing=1e-9,
        alpha=1.0,
        var_prior_weight=0.0,
        weight_level=3,
        l2_reg=1.0,
        max_iter=1000,
        tol=1e-6,
    ):
        super().__init__(
            categorical_features=categorical_features,
            bernoulli_features=bernoulli_features,
            gaussian_features=gaussian_features,
            var_smoothing=var_smoothing,
            alpha=alpha,
            var_prior_weight=var_prior_weight,
        )
        self.weight_level = weight_level
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol

    def _per_factor_log_likelihood_and_keys(self, X):
        """Per-term log-likelihoods of shape (n, n_classes, 1 + n_active) and
        weight keys of shape (n, 1 + n_active) for a validated ``X``."""
        n = X.shape[0]
        n_classes = len(self.classes_)
        n_active = len(self._active_features)
        terms = np.zeros((n, n_classes, 1 + n_active))
        keys = np.zeros((n, 1 + n_active), dtype=np.int64)
        terms[:, :, 0] = np.nan_to_num(self.class_log_prior_, nan=-1e10, neginf=-1e10)
        dims = _LEVEL_DIMS["factor"][self.weight_level]
        by_value = "value" in dims
        pos_of = self._active_pos

        if "gaussian" in self.estimators_:
            indices = self.feature_types_["gaussian"]
            X_gauss = X[:, indices]
            est = self.estimators_["gaussian"]
            diff = X_gauss[:, np.newaxis, :] - est.theta_
            ll = -0.5 * (diff**2 / est.var_ + np.log(2.0 * np.pi * est.var_))
            ll = np.where(np.isnan(X_gauss[:, np.newaxis, :]), 0.0, ll)
            ll = np.nan_to_num(ll, nan=-1e10, neginf=-1e10)
            for k, j in enumerate(indices):
                terms[:, :, 1 + pos_of[j]] = ll[:, :, k]

        if "categorical" in self.estimators_:
            indices = self.feature_types_["categorical"]
            est = self.estimators_["categorical"]
            codes, unseen = self._categorical_codes(X)
            for k, j in enumerate(indices):
                t = 1 + pos_of[j]
                ll = est.feature_log_prob_[k][:, codes[:, k]].T
                terms[:, :, t] = np.nan_to_num(
                    np.where(unseen[:, [k]], 0.0, ll), nan=-1e10, neginf=-1e10
                )
                if by_value:
                    keys[:, t] = np.where(unseen[:, k], -1, codes[:, k])

        if "bernoulli" in self.estimators_:
            indices = self.feature_types_["bernoulli"]
            X_b = X[:, indices]
            missing = np.isnan(X_b)
            bits = (X_b > self.bernoulli_thresholds_).astype(np.int64)
            est = self.estimators_["bernoulli"]
            log_pos = est.feature_log_prob_
            log_neg = np.log1p(-np.exp(log_pos))
            for k, j in enumerate(indices):
                t = 1 + pos_of[j]
                ll = np.where(
                    bits[:, [k]] == 1, log_pos[:, k][None, :], log_neg[:, k][None, :]
                )
                terms[:, :, t] = np.nan_to_num(
                    np.where(missing[:, [k]], 0.0, ll), nan=-1e10, neginf=-1e10
                )
                if by_value:
                    keys[:, t] = np.where(missing[:, k], -1, bits[:, k])

        return terms, keys

    def _setup_weights(self, keys):
        dims = _LEVEL_DIMS["factor"][self.weight_level]
        seen = []
        for t in range(keys.shape[1]):
            col = keys[:, t]
            valid = col[col >= 0]
            seen.append(np.unique(valid) if len(valid) else np.zeros(0, dtype=np.int64))
        self._class_weights = len(self.classes_) if "class" in dims else 1
        sizes = np.array([len(s) for s in seen], dtype=np.int64)
        self._key_off = np.concatenate([[0], np.cumsum(sizes)])
        self._weight_keys = (
            np.concatenate(seen) if sizes.sum() else np.zeros(0, dtype=np.int64)
        )
        self.n_weights_ = int(sizes.sum()) * self._class_weights + 1

    def _weight_indices(self, keys):
        n_seen = self.n_weights_ - 1
        n_classes = len(self.classes_)
        index = np.full(
            keys.shape,
            n_seen,
            dtype=np.int32 if n_seen < 2**31 - n_classes else np.int64,
        )
        for t in range(keys.shape[1]):
            seen = self._weight_keys[self._key_off[t] : self._key_off[t + 1]]
            if not len(seen):
                continue
            pos = np.minimum(np.searchsorted(seen, keys[:, t]), len(seen) - 1)
            offset = self._key_off[t] * self._class_weights
            index[:, t] = np.where(
                seen[pos] == keys[:, t], offset + pos * self._class_weights, n_seen
            )
        return index

    def _scores(self, weights, terms, index):
        n_samples, n_classes, n_terms = terms.shape
        class_step = int(self._class_weights > 1)
        return _weighted_geometric_scores(
            index.ravel(),
            class_step,
            terms.ravel(),
            n_terms,
            _pad_last(weights, class_step * (n_classes - 1)),
            np.zeros((0, 0, 0)),
            np.ones((n_samples, 1), dtype=bool),
            n_classes,
        )

    def _nll_and_grad(self, weights, terms, index, Y):
        scores = self._scores(weights, terms, index)
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        dscores = np.exp(log_prob) * Y.sum(axis=1, keepdims=True) - Y
        n_samples, n_classes, n_terms = terms.shape
        class_step = int(self._class_weights > 1)
        grad = _weighted_geometric_gradient(
            index.ravel(),
            class_step,
            terms.ravel(),
            n_terms,
            dscores,
            np.ones((n_samples, 1), dtype=bool),
            weights.size + class_step * (n_classes - 1),
        )
        with np.errstate(invalid="ignore"):
            nll = -np.sum(np.where(Y > 0, log_prob * Y, 0.0))
        return nll, _fold_padding(grad, weights.size)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes, then learn the discriminative exponent weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors. May contain missing values (``NaN`` or missing
            labels).

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Non-negative weights applied to individual samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X_val, y_val = _validate_input(self, X, y)
        sw = _validate_sample_weight(sample_weight, X_val)
        classes = unique_labels(y_val)
        y_idx = np.searchsorted(classes, y_val)
        if sw is not None and np.any(sw == 0):
            pos = sw > 0
            X_val, y_idx, sw = X_val[pos], y_idx[pos], sw[pos]
        labelled = _label_feature_indices(self)
        self._fit_encoded(X_val, y_idx, classes, labelled, sample_weight=sw)
        self._apply_label_categories(labelled)

        self._active_features = sorted(
            j for cols in self.feature_types_.values() for j in cols
        )
        self._active_pos = {j: m for m, j in enumerate(self._active_features)}
        terms, keys = self._per_factor_log_likelihood_and_keys(X_val)
        self._setup_weights(keys)
        index = self._weight_indices(keys)
        Y = np.eye(len(self.classes_))[y_idx]
        if sw is not None:
            Y = Y * sw[:, None]
        weights, self.n_iter_, converged = _fit_weights(
            lambda w: self._nll_and_grad(w, terms, index, Y),
            np.ones(self.n_weights_),
            self.l2_reg,
            self.max_iter,
            tol=self.tol,
        )
        _warn_not_converged(not converged, 1, self.max_iter)
        self.learned_weights_ = weights
        return self

    def _joint_log_likelihood(self, X):
        check_is_fitted(
            self, attributes=["classes_", "estimators_", "learned_weights_"]
        )
        X = _validate_input(self, X, reset=False)
        n_classes = len(self.classes_)
        n_terms = 1 + len(self._active_features)
        jll = np.empty((X.shape[0], n_classes))
        for rows in _row_chunks(X.shape[0], n_classes * n_terms):
            terms, keys = self._per_factor_log_likelihood_and_keys(X[rows])
            index = self._weight_indices(keys)
            jll[rows] = self._scores(self.learned_weights_, terms, index)
        return jll

    def _term_weights(self, t):
        check_is_fitted(self, attributes=["learned_weights_"])
        lo, hi = self._key_off[t : t + 2] * self._class_weights
        return self.learned_weights_[lo:hi].reshape(-1, self._class_weights)

    @property
    def prior_weights_(self):
        """Weights of the class prior: one per class at ``weight_level=3``,
        a single one below."""
        return self._term_weights(0)[0]

    @property
    def feature_weights_(self):
        """Weights of the conditional of every attribute, a list with one
        array per attribute."""
        check_is_fitted(self, attributes=["learned_weights_"])
        empty = np.empty((0, self._class_weights))
        return [
            (
                self._term_weights(1 + self._active_pos[i])
                if i in self._active_pos
                else empty
            )
            for i in range(self.n_features_in_)
        ]

    def weight_table(self, feature, as_frame=False):
        """Return the weights of an attribute with the values they belong to.

        Parameters
        ----------
        feature : int or str
            The attribute, by its index or by its name in
            ``feature_names_in_``.

        as_frame : bool, default=False
            If True, return a pandas DataFrame indexed by the value of the
            attribute (when ``weight_level >= 2`` and the attribute is
            discrete), with a column per class (or ``weight``).

        Returns
        -------
        table : :class:`~sklearn.utils.Bunch` or DataFrame
            With ``as_frame=False``, a dictionary-like object with the same
            fields as :meth:`WeightedTAN.weight_table`: ``feature``,
            ``weights``, ``feature_values``, ``parents`` (empty),
            ``parent_values`` (empty) and ``classes``.
        """
        idx, kind, pos = self._active_feature(feature)
        m = self._active_pos[idx]
        keys = self._weight_keys[self._key_off[1 + m] : self._key_off[2 + m]]
        dims = _LEVEL_DIMS["factor"][self.weight_level]
        values = None
        if "value" in dims and kind != "gaussian":
            if kind == "categorical":
                values = self.categories_[pos][keys]
            else:
                values = np.array([False, True])[keys]
        table = Bunch(
            feature=idx,
            weights=self._term_weights(1 + m),
            feature_values=values,
            parents=[],
            parent_values=np.empty((len(keys), 0), dtype=object),
            classes=self.classes_ if self._class_weights > 1 else None,
        )
        if not as_frame:
            return table
        index = [] if values is None else [values]
        names = [] if values is None else [_feature_name(self, idx)]
        columns = ["weight"] if table.classes is None else table.classes
        return _table_frame(table.weights, index, names, columns)
