# -*- coding: utf-8 -*-
"""
=====================================================
Handling Mixed Data Types with MixedNB
=====================================================

This example compares three strategies for handling a dataset
with mixed continuous (Gaussian) and discrete (Categorical) features:

1.  **MixedNB**: Our native estimator that models each feature type
    independently.
2.  **OHE Pipeline**: A common but flawed workaround. Categorical features
    are One-Hot Encoded, and the resulting binary features are
    incorrectly modeled by `GaussianNB`.
3.  **Discretizer Pipeline**: The traditional `scikit-learn` workaround.
    Continuous features are binned (discretized) using
    `KBinsDiscretizer`, and the entire dataset is modeled by
    `CategoricalNB`.

As the plots show, `MixedNB` learns a decision boundary that respects the
native data types (a smooth Gaussian curve on the x-axis and sharp
Categorical splits on the y-axis).

The OHE pipeline performs poorly, as the `GaussianNB` fails to model
the 0/1 features. The Discretizer pipeline creates rigid, axis-aligned
boundaries that are an approximation of the true data distribution.

`MixedNB` achieves the best fit (highest log-loss) by avoiding
both information loss (from discretization) and flawed distributional
assumptions (from OHE).
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import DecisionBoundaryDisplay

from skbayes.mixed_nb import MixedNB

# 1. Generate a 2D Mixed-Type dataset
# Feature 0: Gaussian
# Feature 1: Categorical (0, 1, 2)
np.random.seed(42)
n_samples = 300
X = np.zeros((n_samples, 2))
X[:, 0] = np.random.randn(n_samples)  # Gaussian
X[:, 1] = np.random.randint(0, 3, size=n_samples)  # Categorical
X = X.astype(object) # Use object dtype to satisfy ColumnTransformer
X[:, 0] = X[:, 0].astype(float)
X[:, 1] = X[:, 1].astype(int)

# Target 'y' depends on both features
y = (X[:, 0] > 0.5) & (X[:, 1] == 1)
y = y | ((X[:, 0] < -0.5) & (X[:, 1] == 2))
y = y.astype(int)

# --- 2. Define the three models ---

# Model 1: skbayes MixedNB
mnb = MixedNB()
mnb.fit(X, y)

# Model 2: Flawed OHE + GaussianNB Pipeline
preprocessor_ohe = ColumnTransformer(
    [
        ("onehot", OneHotEncoder(drop='first'), [1]),
        ("gauss", "passthrough", [0])
    ],
    remainder="passthrough"
)
pipe_ohe = make_pipeline(preprocessor_ohe, GaussianNB())
pipe_ohe.fit(X, y)

# Model 3: Traditional Discretizer + CategoricalNB Pipeline
preprocessor_kbins = ColumnTransformer(
    [
        ("discretizer", KBinsDiscretizer(n_bins=10, encode='ordinal'), [0]),
        ("categorical", "passthrough", [1])
    ],
    remainder="passthrough"
)
pipe_kbins = make_pipeline(preprocessor_kbins, CategoricalNB())
pipe_kbins.fit(X, y)

models = [mnb, pipe_ohe, pipe_kbins]
titles = [
    "1. MixedNB (Native Hybrid)",
    "2. Pipeline (OHE + GaussianNB)",
    "3. Pipeline (Discretizer + CatNB)"
]

# --- 3. Plot decision boundaries ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, model, title in zip(axes, models, titles):
    DecisionBoundaryDisplay.from_estimator(
        model, X, ax=ax, response_method="predict_proba",
        plot_method="pcolormesh", shading="auto", alpha=0.8,
        # We need to tell the display how to handle object dtype
        # by specifying the features we want to plot
        features=[0, 1], feature_names=["Gaussian", "Categorical"]
    )
    
    # Add jitter to the categorical axis for visualization
    X_plot = X.copy().astype(float)
    X_plot[:, 1] += np.random.rand(n_samples) * 0.4 - 0.2
    
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors="k")
    
    # Calculate and display metrics
    acc = accuracy_score(y, model.predict(X))
    ll = log_loss(y, model.predict_proba(X))
    ax.text(
        0.05, 0.95,
        f"Accuracy: {acc:.3f}\nLog-Loss: {ll:.3f}",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 0 (Continuous)")

axes[0].set_ylabel("Feature 1 (Categorical)")
plt.tight_layout()
fig.suptitle("Comparing MixedNB to scikit-learn Pipeline Workarounds")
plt.subplots_adjust(top=0.85)
plt.show()