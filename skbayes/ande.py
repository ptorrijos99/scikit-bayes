"""
Averaged n-Dependence Estimators (AnDE) implementation utilizing
the 'Super-Class' reduction strategy.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from scipy.special import logsumexp

from .mixed_nb import MixedNB

class AnDE(ClassifierMixin, BaseEstimator):
    """
    Averaged n-Dependence Estimators (AnDE) classifier.

    This implementation leverages the property that an SPnDE (Super-Parent
    n-Dependence Estimator) is mathematically equivalent to a Naive Bayes
    classifier trained on an augmented class variable Y* = (Y, Parents).

    This allows us to reuse the robust `MixedNB` implementation for the
    underlying probability estimation, supporting mixed data types and
    arbitrary dependency orders 'n' without complex custom logic.

    Parameters
    ----------
    n_dependence : int, default=1
        The order of dependence 'n'.
        - n=0: Equivalent to Naive Bayes.
        - n=1: AODE (Averaged One-Dependence Estimators).
        - n=2: A2DE, etc.

    n_bins : int, default=5
        Number of bins for discretizing numerical features ONLY when they
        act as super-parents. Children features remain continuous.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for discretization of super-parents.

    alpha : float, default=1.0
        Smoothing parameter passed to the internal MixedNB estimators.

    Attributes
    ----------
    ensemble_ : list of dict
        Contains the sub-models. Each entry stores:
        - 'parent_indices': tuple of feature indices used as parents.
        - 'estimator': The fitted MixedNB trained on Y*.
        - 'augmented_classes': The classes Y* learned by the estimator.
        - 'enc_map': A mapping to decode Y* back to (Y, Parent_Values).

    classes_ : array-like of shape (n_classes,)
        The unique class labels.
    """

    def __init__(self, n_dependence=1, n_bins=5, strategy='quantile', alpha=1.0):
        self.n_dependence = n_dependence
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha

    def fit(self, X, y):
        """Fit the AnDE model."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # 0. Pre-process Parents: Super-parents must be discrete.
        # Logic: 
        # - If a feature is continuous (float with decimals), discretize it.
        # - If a feature is categorical (int or float without decimals), leave it as is.
        
        self._discretizers_list = {}
        # We work on a copy to create the parent view (integers)
        self._parent_data = np.zeros(X.shape, dtype=int)
        
        # Configuration for discretizer
        kwargs_discretizer = {'subsample': 200_000}
        if self.strategy == 'quantile':
            kwargs_discretizer['quantile_method'] = 'linear'

        for i in range(self.n_features_in_):
            col = X[:, i]
            
            # Detect if column needs discretization
            needs_discretization = False
            
            if np.issubdtype(col.dtype, np.floating):
                # Check if it has non-zero decimals
                # If all mod 1 are 0, it's an integer disguised as float (Categorical)
                if not np.all(np.mod(col, 1) == 0):
                    needs_discretization = True
            
            if needs_discretization:
                # Continuous: Fit discretizer
                est = KBinsDiscretizer(
                    n_bins=self.n_bins, 
                    encode='ordinal', 
                    strategy=self.strategy,
                    **kwargs_discretizer
                )
                # Fit and transform
                self._parent_data[:, i] = est.fit_transform(col.reshape(-1, 1)).flatten()
                self._discretizers_list[i] = est
            else:
                # Discrete (Integer or Float-Integer): Cast to int directly
                self._parent_data[:, i] = col.astype(int)

        # 1. Build Ensemble of SPnDEs (as Augmented Naive Bayes)
        self.ensemble_ = []
        
        # Identify all combinations of n parents
        parent_combinations = list(combinations(range(self.n_features_in_), self.n_dependence))
        
        # If n=0, we just fit a single MixedNB
        if self.n_dependence == 0:
            parent_combinations = [()]

        for parent_indices in parent_combinations:
            # A. Construct the Super-Class Y* = (y, X_parents)
            if self.n_dependence > 0:
                parents_vals = self._parent_data[:, parent_indices]
                # Create a composite label string or tuple
                # We use string for compatibility with LabelEncoder
                # Format: "class_label|p1_val|p2_val..."
                y_augmented = [
                    f"{label}|" + "|".join(map(str, row)) 
                    for label, row in zip(y, parents_vals)
                ]
            else:
                y_augmented = y

            # Encode Y* to integers for MixedNB
            le = LabelEncoder()
            y_augmented_enc = le.fit_transform(y_augmented)
            
            # B. Identify features to keep for the child NB
            # Ideally, X_children = X without parent columns (Eq 16 in paper).
            # The parent probability P(Parents) is absorbed into P(Y*)
            child_indices = [i for i in range(self.n_features_in_) if i not in parent_indices]
            
            if not child_indices:
                # Edge case: n_dependence = n_features (Full Bayesian Network fully connected)
                # In this case, we just learn the Prior P(Y*)
                # We pass a dummy feature of zeros
                X_train_sub = np.zeros((X.shape[0], 1))
            else:
                X_train_sub = X[:, child_indices]

            # C. Fit the Sub-Model (Reuse MixedNB!)
            # We configure MixedNB to auto-detect types of the children
            sub_model = MixedNB(alpha=self.alpha)
            sub_model.fit(X_train_sub, y_augmented_enc)

            # D. Store metadata to decode Y* back to Y during inference
            # We parse the classes_ from the LabelEncoder
            # y_star_classes[k] might be "0|2|5" (Class 0, Parent1=2, Parent2=5)
            decoded_classes = le.inverse_transform(sub_model.classes_)
            
            # Map each internal class index k to (original_class_index, parent_signature_string)
            # We need efficient lookups.
            # Structure: 
            #   map_k_to_y_idx: [0, 0, 1, 1, 0, ...] (original class index for each k)
            #   map_k_to_parent_sig: ["2|5", "1|1", "2|5", ...]
            
            map_k_to_y_idx = []
            map_k_to_parent_sig = []
            
            if self.n_dependence > 0:
                for cls_str in decoded_classes:
                    parts = cls_str.split('|')
                    original_label = type(y[0])(parts[0]) # Cast back to original type (int/str)
                    
                    # Find index in self.classes_
                    y_idx = np.searchsorted(self.classes_, original_label)
                    # If types mismatch searchsorted might fail, robustness check:
                    if self.classes_[y_idx] != original_label:
                         # Fallback if unsorted or types differ
                         y_idx = np.where(self.classes_ == original_label)[0][0]

                    map_k_to_y_idx.append(y_idx)
                    map_k_to_parent_sig.append("|".join(parts[1:]))
            else:
                # n=0 case
                map_k_to_y_idx = np.searchsorted(self.classes_, decoded_classes)
                map_k_to_parent_sig = [""] * len(decoded_classes)

            self.ensemble_.append({
                'parent_indices': parent_indices,
                'child_indices': child_indices,
                'estimator': sub_model,
                'map_y': np.array(map_k_to_y_idx),
                'map_parents': np.array(map_k_to_parent_sig)
            })

        return self

    def _joint_log_likelihood(self, X):
        """
        Compute joint log likelihood P(y, x) aggregated over all models.
        """
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Pre-process parents for the test set (using stored discretizers)
        X_parents_disc = X.copy()
        if hasattr(self, '_discretizers_list'):
            for col_idx, est in self._discretizers_list.items():
                X_parents_disc[:, col_idx] = est.transform(X[:, [col_idx]]).flatten()
        X_parents_disc = X_parents_disc.astype(int)

        # Total log likelihood accumulator
        # We start with 0 probability (log = -inf)
        total_log_probs = np.full((n_samples, n_classes), -np.inf)
        
        n_models = len(self.ensemble_)

        for model_info in self.ensemble_:
            parent_indices = model_info['parent_indices']
            estimator = model_info['estimator']
            child_indices = model_info['child_indices']
            
            # 1. Get raw log probabilities for all Augmented Classes Y*
            # Shape: (n_samples, n_augmented_classes)
            if not child_indices:
                X_test_sub = np.zeros((n_samples, 1))
            else:
                X_test_sub = X[:, child_indices]
                
            # This returns log P(Children | Y*) + log P(Y*) = log P(Children, Y*)
            # Since Y* = (Y, Parents), this is log P(Children, Parents, Y) = log P(X, Y)
            # This is exactly what we want!
            jll_augmented = estimator._joint_log_likelihood(X_test_sub)
            
            if self.n_dependence == 0:
                # Simple case, direct addition
                total_log_probs = np.logaddexp(total_log_probs, jll_augmented)
                continue

            # 2. Filter: Only keep Y* consistent with the observed parents
            # This is the "Super Class" trick inference step.
            
            # Construct signature for current test samples
            # Shape: (n_samples,) e.g. ["0|1", "2|2", ...]
            test_parents_vals = X_parents_disc[:, parent_indices]
            test_signatures = np.array([
                "|".join(map(str, row)) for row in test_parents_vals
            ])
            
            # We need to route the probabilities.
            # Vectorizing this string matching is slightly expensive but clean.
            # Optimization: Broadcast comparison
            
            # map_parents shape: (n_augmented_classes,)
            # test_signatures shape: (n_samples,)
            # matches shape: (n_samples, n_augmented_classes) - Boolean mask
            # Note: If n_samples * n_aug is huge, this might be memory heavy. 
            # But for A1DE/A2DE it's usually manageable.
            
            # Let's do it per augmented class to save memory
            map_y = model_info['map_y']
            map_p = model_info['map_parents']
            
            current_model_probs = np.full((n_samples, n_classes), -np.inf)
            
            for k in range(len(map_p)):
                target_class_idx = map_y[k]
                required_sig = map_p[k]
                
                # Identify samples that have the parents required by this augmented class
                # Using numpy generic string comparison
                valid_mask = (test_signatures == required_sig)
                
                if np.any(valid_mask):
                    # Add the probability to the corresponding original class
                    # But only for the valid samples
                    current_model_probs[valid_mask, target_class_idx] = jll_augmented[valid_mask, k]
            
            # Aggregate this model into the ensemble
            total_log_probs = np.logaddexp(total_log_probs, current_model_probs)

        # Average over models: log(sum(prob)) - log(n_models)
        return total_log_probs - np.log(n_models)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return np.exp(jll - log_prob_x[:, np.newaxis])