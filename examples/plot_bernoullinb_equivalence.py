"""
=====================================================
MixedNB Equivalence with BernoulliNB
=====================================================

This example demonstrates that :class:`skbayes.mixed_nb.MixedNB`
produces identical results to :class:`sklearn.naive_bayes.BernoulliNB`
when all features are binary (Bernoulli).

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
from sklearn.naive_bayes import BernoulliNB

from skbayes.mixed_nb import MixedNB

# 1. Generate a 2D Bernoulli dataset (0s and 1s)
np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 2), dtype=int)
y = (X[:, 0] & X[:, 1]).astype(int)

# 2. Fit both classifiers
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X, y)
probs_bnb = bnb.predict_proba(X)

# MixedNB will auto-detect both features as 'bernoulli' (unique_values=2)
mnb = MixedNB(alpha=1.0)
mnb.fit(X, y)
probs_mnb = mnb.predict_proba(X)

# 3. Assert equivalence
try:
    assert_allclose(probs_bnb, probs_mnb, rtol=1e-7, atol=1e-7)
    equivalence_message = "Probabilities are identical."
except AssertionError as e:
    equivalence_message = f"Probabilities are NOT identical:\n{e}"

print(f"BernoulliNB vs MixedNB Equivalence Check: {equivalence_message}")

# 4. Plot decision boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Create a meshgrid for plotting
xx, yy = np.meshgrid(
    np.arange(-0.5, 2, 1),
    np.arange(-0.5, 2, 1)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# Plot for BernoulliNB
Z_bnb = bnb.predict_proba(grid)[:, 1].reshape(xx.shape)
ax1.pcolormesh(xx, yy, Z_bnb, alpha=0.8, shading="auto", vmin=0, vmax=1)
ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax1.set_title("1. scikit-learn BernoulliNB")
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])

# Plot for MixedNB
Z_mnb = mnb.predict_proba(grid)[:, 1].reshape(xx.shape)
ax2.pcolormesh(xx, yy, Z_mnb, alpha=0.8, shading="auto", vmin=0, vmax=1)
ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax2.set_title("2. skbayes MixedNB (auto-detected)")
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])

fig.suptitle("Equivalence of MixedNB and BernoulliNB on binary data")
plt.show()