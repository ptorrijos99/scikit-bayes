"""
=====================================================
MixedNB Equivalence with CategoricalNB
=====================================================

This example demonstrates that :class:`skbayes.mixed_nb.MixedNB`
produces identical results to :class:`sklearn.naive_bayes.CategoricalNB`
when all features are discrete (categorical).

The plot shows the decision boundaries for both classifiers. As expected,
the boundaries are identical, and the predicted probabilities for the
dataset are all-close.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import CategoricalNB

from skbayes.mixed_nb import MixedNB

# 1. Generate a 2D Categorical dataset (3 categories for f0, 4 for f1)
np.random.seed(42)
X = np.zeros((100, 2), dtype=int)
X[:, 0] = np.random.randint(0, 3, size=100)
X[:, 1] = np.random.randint(0, 4, size=100)
y = (X[:, 0] + X[:, 1] >= 3).astype(int)

# 2. Fit both classifiers
cnb = CategoricalNB(alpha=1.0)
cnb.fit(X, y)
probs_cnb = cnb.predict_proba(X)

# MixedNB will auto-detect both features as 'categorical'
mnb = MixedNB(alpha=1.0)
mnb.fit(X, y)
probs_mnb = mnb.predict_proba(X)

# 3. Assert equivalence
try:
    assert_allclose(probs_cnb, probs_mnb, rtol=1e-7, atol=1e-7)
    equivalence_message = "Probabilities are identical."
except AssertionError as e:
    equivalence_message = f"Probabilities are NOT identical:\n{e}"

print(f"CategoricalNB vs MixedNB Equivalence Check: {equivalence_message}")

# 4. Plot decision boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plotting categorical data requires a meshgrid
xx, yy = np.meshgrid(
    np.arange(-0.5, 3, 1),
    np.arange(-0.5, 4, 1)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# Plot for CategoricalNB
Z_cnb = cnb.predict_proba(grid)[:, 1].reshape(xx.shape)
ax1.pcolormesh(xx, yy, Z_cnb, alpha=0.8, shading="auto")
ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax1.set_title("1. scikit-learn CategoricalNB")
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2, 3])

# Plot for MixedNB
Z_mnb = mnb.predict_proba(grid)[:, 1].reshape(xx.shape)
ax2.pcolormesh(xx, yy, Z_mnb, alpha=0.8, shading="auto")
ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax2.set_title("2. skbayes MixedNB (auto-detected)")
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2, 3])

fig.suptitle("Equivalence of MixedNB and CategoricalNB on discrete data")
plt.show()