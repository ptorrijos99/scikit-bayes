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

`MixedNB` achieves the best fit by avoiding
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

from skbayes.mixed_nb import MixedNB

# --- 1. Generate a 2D Mixed-Type dataset
# Feature 0: Gaussian
# Feature 1: Categorical (0, 1, 2)
np.random.seed(42)
n_samples = 300
X_num = np.zeros((n_samples, 2))
X_num[:, 0] = np.random.randn(n_samples)  # Gaussian
X_num[:, 1] = np.random.randint(0, 3, size=n_samples)  # Categorical

# Target 'y' depends on both features
y = (X_num[:, 0] > 0.5) & (X_num[:, 1] == 1)
y = y | ((X_num[:, 0] < -0.5) & (X_num[:, 1] == 2))
y = y.astype(int)

# --- 2. Define the three models ---
# Create X_obj for scikit-learn pipelines that require object dtype for ColumnTransformer
X_obj = X_num.astype(object)

# Model 1: skbayes MixedNB (fits on numeric data)
mnb = MixedNB()
mnb.fit(X_num, y)

# Model 2: Flawed OHE + GaussianNB Pipeline
preprocessor_ohe = ColumnTransformer(
    [
        ("onehot", OneHotEncoder(drop='first', handle_unknown='ignore'), [1]),
        ("gauss", "passthrough", [0])
    ],
    remainder="passthrough"
)
pipe_ohe = make_pipeline(preprocessor_ohe, GaussianNB())
pipe_ohe.fit(X_obj, y)

# Model 3: Traditional Discretizer + CategoricalNB Pipeline
preprocessor_kbins = ColumnTransformer(
    [
        ("discretizer", KBinsDiscretizer(n_bins=10, encode='ordinal'), [0]),
        ("categorical", "passthrough", [1])
    ],
    remainder="passthrough"
)
pipe_kbins = make_pipeline(preprocessor_kbins, CategoricalNB())
pipe_kbins.fit(X_obj, y)

models = [mnb, pipe_ohe, pipe_kbins]
titles = [
    "1. MixedNB (Native Hybrid)",
    "2. Pipeline (OHE + GaussianNB)",
    "3. Pipeline (Discretizer + CatNB)"
]

# --- 3. Plot decision boundaries and metrics ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Create a meshgrid for plotting
h = 0.05  # step size in the mesh
x_min, x_max = X_num[:, 0].min() - 1, X_num[:, 0].max() + 1
y_min, y_max = X_num[:, 1].min() - 0.5, X_num[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, 1))

# Plot for each model
for ax, model, title in zip(axes, models, titles):
    # Prepare the grid for prediction based on model type
    grid_num = np.c_[xx.ravel(), yy.ravel()]
    if title == titles[0]: # MixedNB
        Z = model.predict_proba(grid_num)[:, 1]
    else: # Pipelines
        grid_obj = grid_num.astype(object)
        Z = model.predict_proba(grid_obj)[:, 1]

    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, alpha=0.8, shading="auto", vmin=0, vmax=1)
    
    # Add jitter to the categorical axis for visualization
    X_plot = X_num.copy()
    X_plot[:, 1] += np.random.rand(n_samples) * 0.4 - 0.2
    
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors="k")
    
    # Calculate and display metrics
    X_fit = X_num if title == titles[0] else X_obj
    acc = accuracy_score(y, model.predict(X_fit))
    ll = log_loss(y, model.predict_proba(X_fit))
    ax.text(
        0.05, 0.95,
        f"Accuracy: {acc:.3f}\nLog-Loss: {ll:.3f}",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 0 (Continuous)")
    ax.set_yticks([0, 1, 2]) # Set categorical ticks

axes[0].set_ylabel("Feature 1 (Categorical)")
plt.tight_layout()
fig.suptitle("Comparing MixedNB to scikit-learn Pipeline Workarounds")
plt.subplots_adjust(top=0.85)
plt.show()
