"""Naive Bayes classifier for heterogeneous (mixed-type) data."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MixedNB(BaseEstimator, ClassifierMixin):
    """
    Mixed Naive Bayes classifier for heterogeneous data.

    This classifier is designed to handle datasets with a mix of continuous
    (Gaussian), categorical, and binary (Bernoulli) features. It internally
    uses scikit-learn's GaussianNB, CategoricalNB, and BernoulliNB on the
    respective feature subsets.

    Features are classified and handled as follows:
    1.  User-defined: The user can explicitly specify which features are
        categorical or Bernoulli using the constructor parameters.
    2.  Auto-detection: For features not specified by the user, the classifier
        will attempt to infer the type during `fit`:
        -   Features with exactly two unique values are treated as Bernoulli.
        -   Integer features with more than two unique values are treated as
            Categorical. Note: These features must be encoded as non-negative
            integers (0, 1, 2, ...).
        -   Floating-point features are treated as Gaussian.

    Read more in the :ref:`User Guide <mixed_naive_bayes>`.

    Parameters
    ----------
    categorical_features : array-like of shape (n_categorical_features,), default=None
        A list of indices for the features that should be treated as
        categorical. If None, categorical features are inferred from data
        where dtype is integer and the number of unique values is greater
        than 2.
        .. warning::
           Categorical features must be encoded as non-negative integers.

    bernoulli_features : array-like of shape (n_bernoulli_features,), default=None
        A list of indices for the features that should be treated as Bernoulli.
        If None, Bernoulli features are inferred from data where the number
        of unique values is exactly 2.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all Gaussian features that is added
        to variances for calculation stability. Passed to `GaussianNB`.

    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter. Passed to
        `CategoricalNB` and `BernoulliNB`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    feature_types_ : dict
        A dictionary mapping feature type ('gaussian', 'categorical',
        'bernoulli') to the indices of the features of that type.

    estimators_ : dict
        A dictionary containing the fitted Naive Bayes estimators for each
        feature type.

    See Also
    --------
    GaussianNB : Naive Bayes classifier for Gaussian features.
    CategoricalNB : Naive Bayes classifier for categorical features.
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.

    Examples
    --------
    >>> import numpy as np
    >>> from skbayes.mixed_nb import MixedNB
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
    def __init__(self, categorical_features=None, bernoulli_features=None,
                 var_smoothing=1e-9, alpha=1.0):
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def _validate_feature_indices(self, indices, n_features, name):
        """Validate provided feature indices."""
        if indices is None:
            return []
        indices = np.asarray(indices, dtype=int)
        if indices.ndim > 1:
            raise ValueError(f"'{name}' must be a 1D array-like.")
        if np.any(indices < 0) or np.any(indices >= n_features):
            raise ValueError(
                f"All indices in '{name}' must be in [0, {n_features - 1}]."
            )
        return indices.tolist()

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
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        # --- Parameter and Feature Type Validation ---
        cat_feats = self._validate_feature_indices(
            self.categorical_features, self.n_features_in_, "categorical_features"
        )
        bern_feats = self._validate_feature_indices(
            self.bernoulli_features, self.n_features_in_, "bernoulli_features"
        )

        if set(cat_feats) & set(bern_feats):
            raise ValueError(
                "The sets of categorical and Bernoulli features must be disjoint."
            )

        self.feature_types_ = {
            'gaussian': [], 'categorical': cat_feats, 'bernoulli': bern_feats
        }
        
        user_defined_indices = set(cat_feats) | set(bern_feats)

        # Auto-detect remaining features
        for i in range(self.n_features_in_):
            if i in user_defined_indices:
                continue
            
            feature_col = X[:, i]
            unique_vals = np.unique(feature_col)

            if len(unique_vals) == 1:
                # Constant feature, can be ignored
                continue
            elif len(unique_vals) == 2:
                self.feature_types_['bernoulli'].append(i)
            elif np.issubdtype(feature_col.dtype, np.integer):
                self.feature_types_['categorical'].append(i)
            else: # np.issubdtype(feature_col.dtype, np.floating)
                self.feature_types_['gaussian'].append(i)

        # --- Fit Sub-estimators ---
        self.estimators_ = {}
        
        if self.feature_types_['gaussian']:
            indices = self.feature_types_['gaussian']
            gauss_nb = GaussianNB(var_smoothing=self.var_smoothing)
            gauss_nb.fit(X[:, indices], y)
            self.estimators_['gaussian'] = gauss_nb

        if self.feature_types_['categorical']:
            indices = self.feature_types_['categorical']
            cat_nb = CategoricalNB(alpha=self.alpha, min_categories=self._get_min_categories(X[:, indices]))
            cat_nb.fit(X[:, indices], y)
            self.estimators_['categorical'] = cat_nb

        if self.feature_types_['bernoulli']:
            indices = self.feature_types_['bernoulli']
            bern_nb = BernoulliNB(alpha=self.alpha)
            # BernoulliNB works on binary data, binarize might be needed if not 0/1
            X_bern = X[:, indices] > 0
            bern_nb.fit(X_bern, y)
            self.estimators_['bernoulli'] = bern_nb
            
        if self.estimators_:
            any_estimator = next(iter(self.estimators_.values()))
            self.class_log_prior_ = any_estimator.class_log_prior_
        else:
            le = LabelEncoder().fit(y)
            class_counts = np.bincount(le.transform(y))
            self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        return self

    def _get_min_categories(self, X_cat):
        """Helper to determine min_categories for CategoricalNB."""
        if X_cat.shape[1] == 0:
            return None
        return (X_cat.max(axis=0) + 1).astype(int, copy=False)

    def _joint_log_likelihood(self, X):
        """Calculate the unnormalized posterior log probability of X."""
        check_is_fitted(self)
        X = check_array(X)
        
        jll = np.zeros((X.shape[0], len(self.classes_)))

        # Gaussian features
        if 'gaussian' in self.estimators_:
            indices = self.feature_types_['gaussian']
            X_gauss = X[:, indices]  # shape: (n_samples, n_gauss)
            est = self.estimators_['gaussian']
            # Compute log P(x|c) for each class i: sum over features of ((x - mu_i)^2 / var_i)
            diff = X_gauss[:, np.newaxis, :] - est.theta_  # (n_samples, n_classes, n_gauss)
            log_prob = -0.5 * np.sum((diff ** 2) / est.var_, axis=2)  # (n_samples, n_classes)
            log_constant = -0.5 * np.sum(np.log(2.0 * np.pi * est.var_), axis=1)  # (n_classes,)
            jll += log_prob + log_constant  # broadcasting: (n_samples, n_classes) + (n_classes,)
    
        # Categorical features
        if 'categorical' in self.estimators_:
            indices = self.feature_types_['categorical']
            X_cat = X[:, indices].astype(int)
            est = self.estimators_['categorical']
            # log P(x_i|c) is stored in feature_log_prob_
            for i in range(X_cat.shape[1]):
                jll += est.feature_log_prob_[i][:, X_cat[:, i]].T
        
        # Bernoulli features
        if 'bernoulli' in self.estimators_:
            indices = self.feature_types_['bernoulli']
            X_bern = X[:, indices] > 0
            est = self.estimators_['bernoulli']
            # log P(x_i|c) for Bernoulli
            log_prob_pos = est.feature_log_prob_
            log_prob_neg = np.log(1 - np.exp(log_prob_pos))
            jll += X_bern @ (log_prob_pos - log_prob_neg).T
            jll += np.sum(log_prob_neg, axis=1)

        return jll + self.class_log_prior_

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
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]