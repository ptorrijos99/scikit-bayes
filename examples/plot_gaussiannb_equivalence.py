"""
=====================================================
MixedNB Equivalence with GaussianNB
=====================================================

This example demonstrates that :class:`skbayes.mixed_nb.MixedNB`
produces identical results to :class:`sklearn.naive_bayes.GaussianNB`
when all features are continuous (Gaussian).

The plot shows the decision boundaries for both classifiers. As expected,
the boundaries are identical, and the predicted probabilities for the
dataset are all-close.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB

from skbayes.mixed_nb import MixedNB

# 1. Generate a 2D Gaussian dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# 2. Fit both classifiers
gnb = GaussianNB()
gnb.fit(X, y)
probs_gnb = gnb.predict_proba(X)

# MixedNB will auto-detect both features as 'gaussian'
mnb = MixedNB()
mnb.fit(X, y)
probs_mnb = mnb.predict_proba(X)

# 3. Assert equivalence
try:
    assert_allclose(probs_gnb, probs_mnb, rtol=1e-7, atol=1e-7)
    equivalence_message = "Probabilities are identical."
except AssertionError as e:
    equivalence_message = f"Probabilities are NOT identical:\n{e}"

print(f"GaussianNB vs MixedNB Equivalence Check: {equivalence_message}")

# 4. Plot decision boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

DecisionBoundaryDisplay.from_estimator(
    gnb, X, ax=ax1, response_method="predict_proba",
    plot_method="pcolormesh", shading="auto", alpha=0.8
)
ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax1.set_title("1. scikit-learn GaussianNB")

DecisionBoundaryDisplay.from_estimator(
    mnb, X, ax=ax2, response_method="predict_proba",
    plot_method="pcolormesh", shading="auto", alpha=0.8
)
ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax2.set_title("2. skbayes MixedNB (auto-detected)")

fig.suptitle("Equivalence of MixedNB and GaussianNB on continuous data")
plt.show()