"""
Family of Averaged n-Dependence Estimators (AnDE) and Accelerated Logistic Regression (ALR).
Supports mixed data types via the 'Super-Class' strategy.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, LabelBinarizer

from .mixed_nb import MixedNB

class _BaseAnDE(ClassifierMixin, BaseEstimator):
    """
    Base class for the AnDE family of algorithms.
    
    This class implements the **Generative Phase** using the **"Super-Class" (or Augmented Class) 
    fitting strategy**. It serves as the foundation for both AnDE (Arithmetic Mean) 
    and ALR/AnJE (Geometric Mean).

    **Mathematical Formulation:**
    
    An SPnDE (Super-Parent n-Dependence Estimator) models the joint probability 
    $P(y, \\mathbf{x})$ by conditioning all attributes on the class $y$ and a subset 
    of parent attributes $\\mathbf{x}_p$ (where $|\\mathbf{x}_p| = n$).
    
    .. math::
        P(y, \\mathbf{x}) = P(y, \\mathbf{x}_p) \\prod_{i \\notin p} P(x_i \\mid y, \\mathbf{x}_p)

    By defining an **Augmented Super-Class** $Y^* = (Y, \\mathbf{X}_p)$, which represents 
    the Cartesian product of the original class and the unique values of the super-parents, 
    the equation simplifies to a standard Naive Bayes structure:

    .. math::
        P(Y^*, \\mathbf{x}_{children}) = P(Y^*) \\prod_{i \\in children} P(x_i \\mid Y^*)

    **Implementation Details:**
    
    1.  **Parent Discretization:** Since $Y^*$ acts as a discrete class identifier, 
        the parent attributes $\\mathbf{X}_p$ must be discrete. This class automatically 
        detects continuous parents and discretizes them (using `KBinsDiscretizer`), 
        while leaving categorical parents untouched.
    
    2.  **Delegate to MixedNB:** Once $Y^*$ is constructed, the problem of estimating 
        $P(x_i \\mid Y^*)$ becomes a standard Naive Bayes problem. We delegate this 
        to `MixedNB`, which natively handles Gaussian (continuous), Bernoulli (binary), 
        and Categorical (discrete) children distributions.
    
    3.  **Ensemble Storage:** It stores a list of sub-models (one for each combination 
        of parents), including the mapping logic to decode predictions from $Y^*$ 
        back to the original class space $Y$.

    Parameters
    ----------
    n_dependence : int, default=1
        Order of dependence 'n'.
    
    n_bins : int, default=5
        Number of bins for discretizing numerical features when they act as super-parents.
    
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used for discretization.
    
    alpha : float, default=1.0
        Smoothing parameter for the internal MixedNB estimators.
    """
    def __init__(self, n_dependence=1, n_bins=5, strategy='quantile', alpha=1.0):
        self.n_dependence = n_dependence
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha

    def fit(self, X, y):
        """
        Generative fitting.
        Learns the joint probability P(y, x) for each subspace (SPODE) by counting frequencies.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # --- 1. Discretize Parents (Super-Class logic) ---
        self._discretizers_list = {}
        self._parent_data = np.zeros(X.shape, dtype=int)
        
        # Prepare kwargs for discretizer to avoid warnings
        kwargs_discretizer = {'subsample': 200_000}
        if self.strategy == 'quantile':
            kwargs_discretizer['quantile_method'] = 'linear'

        for i in range(self.n_features_in_):
            col = X[:, i]
            
            # Smart check: Float but effectively integer?
            is_continuous = False
            if np.issubdtype(col.dtype, np.floating):
                if not np.all(np.mod(col, 1) == 0):
                    is_continuous = True
            
            if is_continuous:
                est = KBinsDiscretizer(
                    n_bins=self.n_bins, 
                    encode='ordinal', 
                    strategy=self.strategy, 
                    **kwargs_discretizer
                )
                self._parent_data[:, i] = est.fit_transform(col.reshape(-1, 1)).flatten()
                self._discretizers_list[i] = est
            else:
                self._parent_data[:, i] = col.astype(int)

        # --- 2. Build Ensemble (Generative) ---
        self.ensemble_ = []
        parent_combinations = list(combinations(range(self.n_features_in_), self.n_dependence))
        
        # n=0 case (Naive Bayes)
        if self.n_dependence == 0:
            parent_combinations = [()]

        for parent_indices in parent_combinations:
            # A. Construct Y* = (y, X_parents)
            if self.n_dependence > 0:
                parents_vals = self._parent_data[:, parent_indices]
                # Efficient string signature construction
                y_augmented = [
                    f"{label}|" + "|".join(map(str, row)) 
                    for label, row in zip(y, parents_vals)
                ]
            else:
                y_augmented = y

            le = LabelEncoder()
            y_augmented_enc = le.fit_transform(y_augmented)
            
            # B. Identify features for child NB (all except parents)
            child_indices = [i for i in range(self.n_features_in_) if i not in parent_indices]
            
            if not child_indices:
                X_train_sub = np.zeros((X.shape[0], 1)) # Dummy feature if fully connected
            else:
                X_train_sub = X[:, child_indices]

            # C. Fit the Sub-Model (Reuse MixedNB)
            sub_model = MixedNB(alpha=self.alpha)
            sub_model.fit(X_train_sub, y_augmented_enc)

            # D. Store metadata for decoding Y* -> Y
            decoded_classes = le.inverse_transform(sub_model.classes_)
            map_k_to_y_idx = []
            map_k_to_parent_sig = []
            
            for cls_str in decoded_classes:
                if self.n_dependence > 0:
                    parts = cls_str.split('|')
                    # Robust type casting back to original label type
                    original_label = type(y[0])(parts[0]) 
                    parent_sig = "|".join(parts[1:])
                else:
                    original_label = cls_str
                    parent_sig = ""
                
                # Find index in self.classes_
                y_idx = np.where(self.classes_ == original_label)[0][0]
                
                map_k_to_y_idx.append(y_idx)
                map_k_to_parent_sig.append(parent_sig)

            self.ensemble_.append({
                'parent_indices': parent_indices,
                'child_indices': child_indices,
                'estimator': sub_model,
                'map_y': np.array(map_k_to_y_idx),
                'map_parents': np.array(map_k_to_parent_sig)
            })
            
        return self

    def _get_jll_per_model(self, X):
        """
        Computes the log-probability P(y, x | m) for each model 'm' in the ensemble.
        
        Returns
        -------
        jll_tensor : ndarray of shape (n_samples, n_classes, n_models)
            Contains log P(y, x) according to each SPODE.
            If a SPODE doesn't cover a sample (mismatch parents), returns -inf.
        """
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_models = len(self.ensemble_)
        
        # Discretize parents for test set
        X_parents_disc = X.copy()
        if hasattr(self, '_discretizers_list'):
            for col_idx, est in self._discretizers_list.items():
                X_parents_disc[:, col_idx] = est.transform(X[:, [col_idx]]).flatten()
        X_parents_disc = X_parents_disc.astype(int)

        # Output tensor initialized to log(0) = -inf
        jll_tensor = np.full((n_samples, n_classes, n_models), -np.inf)

        for m_idx, model_info in enumerate(self.ensemble_):
            parent_indices = model_info['parent_indices']
            estimator = model_info['estimator']
            child_indices = model_info['child_indices']
            map_y = model_info['map_y']
            map_p = model_info['map_parents']

            # 1. Get raw log probabilities for Y* (Augmented Classes)
            if not child_indices:
                X_test_sub = np.zeros((n_samples, 1))
            else:
                X_test_sub = X[:, child_indices]
            
            # This returns log P(Children, Y*)
            jll_augmented = estimator._joint_log_likelihood(X_test_sub)

            if self.n_dependence == 0:
                jll_tensor[:, :, m_idx] = jll_augmented
                continue

            # 2. Filter: Map Y* back to Y only where Parents match
            test_parents_vals = X_parents_disc[:, parent_indices]
            test_signatures = np.array([
                "|".join(map(str, row)) for row in test_parents_vals
            ])

            # Iterate over augmented classes to route probabilities
            # This mapping step implements the "Super-Class" inference logic
            for k in range(len(map_p)):
                target_class_idx = map_y[k]
                required_sig = map_p[k]
                
                # Check signature match
                valid_mask = (test_signatures == required_sig)
                
                if np.any(valid_mask):
                    jll_tensor[valid_mask, target_class_idx, m_idx] = jll_augmented[valid_mask, k]
        
        return jll_tensor


# =============================================================================
# 1. Generative Families (Classic AnDE)
# =============================================================================

class AnDE(_BaseAnDE):
    """
    Averaged n-Dependence Estimators (AnDE).
    
    This is the standard generative model (e.g., AODE for n=1, A2DE for n=2).
    It aggregates the predictions of sub-models (SPODEs) using an **Arithmetic Mean**.
    
    .. math::
        P(y|x) \\propto \\sum_{i} P_i(y, x)

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
        Contains the sub-models. Each entry stores the 'estimator' (MixedNB)
        and mapping metadata for the super-class strategy.

    classes_ : array-like of shape (n_classes,)
        The unique class labels.
    """
    def predict_log_proba(self, X):
        jll_models = self._get_jll_per_model(X) 
        
        # Arithmetic Mean in Log Space: log(mean(exp(jll)))
        # = logsumexp(jll) - log(M)
        total_jll = logsumexp(jll_models, axis=2) - np.log(jll_models.shape[2])
        
        # Normalize to posterior P(y|x)
        log_prob_x = logsumexp(total_jll, axis=1)
        return total_jll - log_prob_x[:, np.newaxis]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class AnJE(_BaseAnDE):
    """
    Averaged n-Join Estimators (AnJE).
    
    A generative model similar to AnDE, but aggregates using a **Geometric Mean**.
    
    .. math::
        P(y|x) \\propto \\prod_{i} P_i(y, x) \\equiv \\exp \\left( \\sum_{i} \\log P_i(y, x) \\right)
    
    This model is less common on its own but serves as the initialization
    basis for Accelerated Logistic Regression (ALR).

    Parameters
    ----------
    n_dependence : int, default=1
        The order of dependence.

    n_bins : int, default=5
        Number of bins for discretizing super-parents.

    strategy : str, default='quantile'
        Discretization strategy.

    alpha : float, default=1.0
        Smoothing parameter.
    """
    def predict_log_proba(self, X):
        jll_models = self._get_jll_per_model(X)
        
        # Geometric Mean in Log Space = Sum
        total_jll = np.sum(jll_models, axis=2)
        
        # Normalize
        log_prob_x = logsumexp(total_jll, axis=1)
        return total_jll - log_prob_x[:, np.newaxis]
        
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# =============================================================================
# 2. Discriminative / Hybrid Families (Learned Weights)
# =============================================================================

class ALR(AnJE):
    """
    Accelerated Logistic Regression (ALR).
    
    A hybrid generative-discriminative classifier. It starts with the AnJE 
    generative solution and refines it by learning a weight for each sub-model
    (SPODE) to maximize Conditional Log-Likelihood (CLL).
    
    Because it uses the Geometric Mean structure (log-linear), the optimization 
    problem is **CONVEX**, guaranteeing a global optimum and fast convergence
    using L-BFGS-B.
    
    .. math::
        \\log P(y|x) \\propto \\sum_{i} w_i \\cdot \\log P_i(y, x)

    Parameters
    ----------
    n_dependence : int, default=1
        Order of dependence.

    n_bins : int, default=5
        Bins for super-parents.

    strategy : str, default='quantile'
        Discretization strategy.

    alpha : float, default=1.0
        Smoothing for the generative phase.

    l2_reg : float, default=1e-4
        L2 regularization strength for the weights during the discriminative phase.

    max_iter : int, default=100
        Maximum iterations for the L-BFGS-B optimizer.

    Attributes
    ----------
    weights_ : ndarray of shape (n_models,)
        The learned discriminative weights for each SPODE in the ensemble.
    """
    def __init__(self, n_dependence=1, n_bins=5, strategy='quantile', alpha=1.0, 
                 l2_reg=1e-4, max_iter=100):
        super().__init__(n_dependence, n_bins, strategy, alpha)
        self.l2_reg = l2_reg
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the ALR model.
        
        1. Runs generative fitting (frequency counting).
        2. Optimizes weights 'w' discriminatively to maximize CLL.
        """
        # 1. Generative Phase (Pre-conditioning)
        super().fit(X, y)
        
        # 2. Discriminative Phase (Optimization)
        X_check = check_array(X)
        jll_tensor = self._get_jll_per_model(X_check) # (N, C, M)
        jll_tensor = np.clip(jll_tensor, -1e10, 700)
        
        # Prepare Target (One-Hot)
        lb = LabelBinarizer()
        y_ohe = lb.fit_transform(y)
        if len(self.classes_) == 2:
            y_ohe = np.hstack((1 - y_ohe, y_ohe))

        def objective(weights):
            # weights: (M,)
            # Linear combination: Sum_m (w_m * JLL_m)
            weighted_jll = jll_tensor * weights.reshape(1, 1, -1)

            # Handle 0 * -inf = nan.
            weighted_jll = np.nan_to_num(weighted_jll, nan=0.0)

            final_jll = np.sum(weighted_jll, axis=2)

            # Clip to prevent overflow in exp/logsumexp
            final_jll = np.clip(final_jll, -1e10, 700)
            
            # Softmax normalization (LogSumExp)
            lse = logsumexp(final_jll, axis=1)
            log_proba = final_jll - lse[:, np.newaxis]
            
            # Neg Log Likelihood
            true_class_log_probs = log_proba[y_ohe.astype(bool)]
            nll = -np.sum(true_class_log_probs)
            
            # L2 Regularization
            reg = self.l2_reg * np.sum(weights**2)
            return nll + reg

        # Optimize
        n_models = len(self.ensemble_)
        # Initialize with 1.0 (Generative solution)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, None) for _ in range(n_models)]
        
        # Suppress warnings during optimization loop
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            res = minimize(
                objective, initial_weights, method='L-BFGS-B', 
                bounds=bounds, options={'maxiter': self.max_iter}
            )
        
        self.weights_ = res.x
        return self

    def predict_log_proba(self, X):
        jll_models = self._get_jll_per_model(X)
        jll_models = np.clip(jll_models, -1e10, 700)
        
        # Weighted Geometric Mean
        w = self.weights_.reshape(1, 1, -1)
        weighted_terms = jll_models * w
        
        total_jll = np.sum(weighted_terms, axis=2)
        jll_models = np.clip(jll_models, -1e10, 700)

        log_prob_x = logsumexp(total_jll, axis=1)
        return total_jll - log_prob_x[:, np.newaxis]


class WeightedAnDE(AnDE):
    """
    Weighted AnDE (Hybrid).
    
    A discriminative weighting of standard AnDE (Arithmetic Mean).
    
    .. math::
        P(y|x) \\propto \\sum_{i} w_i \\cdot P_i(y, x)
    
    Note: Unlike ALR, this optimization problem is **NON-CONVEX**. 
    L-BFGS-B may find local minima, but initializing with w=1 (Generative solution)
    usually yields good results.

    Parameters
    ----------
    n_dependence : int, default=1
        Order of dependence.

    l2_reg : float, default=1e-4
        L2 regularization.

    max_iter : int, default=100
        Maximum iterations.
    """
    def __init__(self, n_dependence=1, n_bins=5, strategy='quantile', alpha=1.0, 
                 l2_reg=1e-4, max_iter=100):
        super().__init__(n_dependence, n_bins, strategy, alpha)
        self.l2_reg = l2_reg
        self.max_iter = max_iter

    def fit(self, X, y):
        # 1. Generative Phase
        super().fit(X, y)
        
        # 2. Discriminative Phase
        X_check = check_array(X)
        jll_tensor = self._get_jll_per_model(X_check) # (N, C, M)
        jll_tensor = np.clip(jll_tensor, -1e10, 700)
        
        lb = LabelBinarizer()
        y_ohe = lb.fit_transform(y)
        if len(self.classes_) == 2:
            y_ohe = np.hstack((1 - y_ohe, y_ohe))

        # Objective Function (Non-Convex)
        def objective(weights):
            # weights: (M,)
            # Arithmetic Mean: log( Sum_m (w_m * exp(JLL_m)) )
            # = logsumexp( log(w) + JLL )
            
            # Avoid log(0) for weights
            w_safe = np.maximum(weights, 1e-10)
            log_w = np.log(w_safe).reshape(1, 1, -1)
            
            # Combine
            weighted_log_terms = log_w + jll_tensor

            weighted_log_terms = np.clip(weighted_log_terms, -1e10, 700)

            final_jll = logsumexp(weighted_log_terms, axis=2)
            final_jll = np.clip(final_jll, -1e10, 700)
            
            # Softmax
            lse = logsumexp(final_jll, axis=1)
            log_proba = final_jll - lse[:, np.newaxis]
            
            # Neg Log Likelihood
            true_class_log_probs = log_proba[y_ohe.astype(bool)]
            nll = -np.sum(true_class_log_probs)
            
            # L2 Regularization
            reg = self.l2_reg * np.sum(weights**2)
            return nll + reg

        n_models = len(self.ensemble_)
        # Init with 1.0 (Generative solution)
        initial_weights = np.ones(n_models)
        bounds = [(0, None) for _ in range(n_models)]

        # Suppress warnings during optimization loop
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            res = minimize(
                objective, initial_weights, method='L-BFGS-B', 
                bounds=bounds, options={'maxiter': self.max_iter}
            )
        
        self.weights_ = res.x
        return self

    def predict_log_proba(self, X):
        jll_models = self._get_jll_per_model(X)
        
        # Weighted Arithmetic Mean
        w = np.maximum(self.weights_, 1e-10).reshape(1, 1, -1)
        
        # log( sum( w * exp(jll) ) )
        weighted_log_terms = np.log(w) + jll_models

        weighted_log_terms = np.clip(weighted_log_terms, -700, 700)
        total_jll = logsumexp(weighted_log_terms, axis=2)
        
        log_prob_x = logsumexp(total_jll, axis=1)
        return total_jll - log_prob_x[:, np.newaxis]