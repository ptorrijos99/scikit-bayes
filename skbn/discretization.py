
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class DecisionTreeDiscretizer(BaseEstimator, TransformerMixin):
    """
    Supervised discretizer that uses Decision Trees to find optimal binning boundaries.
    This serves as a proxy for MDL (Fayyad & Irani) discretization.
    
    Parameters
    ----------
    n_bins : int, default=10
        Maximum number of bins per feature.
    criterion : {'gini', 'entropy'}, default='entropy'
        The function to measure the quality of a split. 'entropy' is closer to MDL.
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf node.
    """
    def __init__(self, n_bins=10, criterion='entropy', min_samples_leaf=5, random_state=None):
        self.n_bins = n_bins
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.binner_ = {}

    def fit(self, X, y):
        """
        Fits a Decision Tree for each feature to determine cut points.
        """
        X, y = check_X_y(X, y, force_all_finite='allow-nan')
        self.n_features_in_ = X.shape[1]
        self.binner_ = {}
        
        for i in range(self.n_features_in_):
            # Fit tree on single feature
            dt = DecisionTreeClassifier(
                criterion=self.criterion,
                max_leaf_nodes=self.n_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            # Handle NaNs? sklearn trees don't support them well yet (depending on version).
            # For robustness, we impute with median for the discretization step only.
            xi = X[:, i].reshape(-1, 1)
            
            # Simple imputation if needed
            if np.isnan(xi).any():
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(strategy='median')
                xi = imp.fit_transform(xi)
                
            dt.fit(xi, y)
            
            # Extract thresholds (non-leaf nodes)
            target_thresholds = dt.tree_.threshold[dt.tree_.threshold != -2]
            target_thresholds = np.sort(np.unique(target_thresholds))
            
            # Boundaries: [-inf, t1, t2, ..., inf]
            self.binner_[i] = target_thresholds
            
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, force_all_finite='allow-nan')
        
        X_out = X.copy()
        
        for i in range(self.n_features_in_):
            if i in self.binner_:
                thresholds = self.binner_[i]
                if len(thresholds) > 0:
                    # np.digitize returns indices of bins to which each value belongs
                    # bins[i-1] <= x < bins[i]
                    X_out[:, i] = np.digitize(X[:, i], thresholds)
                else:
                    X_out[:, i] = 0 # No splits found, all one bin
                    
        return X_out
