"""
=====================================================
MixedNB equivalence with GaussianNB
=====================================================

When every feature is continuous, :class:`skbn.MixedNB` detects them as
Gaussian and fits the same model as :class:`sklearn.naive_bayes.GaussianNB`.
Both classifiers are fitted here on a two-dimensional dataset with two
informative features, and their predicted probabilities are compared. The
plot shows the probability of class 1 over the feature plane for each
classifier; the two maps coincide.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB

from skbn import MixedNB

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

gnb = GaussianNB()
gnb.fit(X, y)
probs_gnb = gnb.predict_proba(X)

# MixedNB detects both features as 'gaussian'
mnb = MixedNB()
mnb.fit(X, y)
probs_mnb = mnb.predict_proba(X)

try:
    assert_allclose(probs_gnb, probs_mnb, rtol=1e-7, atol=1e-7)
    equivalence_message = "Probabilities are identical."
except AssertionError as e:
    equivalence_message = f"Probabilities are NOT identical:\n{e}"

print(f"GaussianNB vs MixedNB: {equivalence_message}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

models = [gnb, mnb]
titles = ["GaussianNB (scikit-learn)", "MixedNB (skbn)"]

for ax, model, title in zip(axes, models, titles):
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        ax=ax,
        response_method="predict_proba",
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.8,
        cmap="viridis",
    )

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="indigo",
        marker="o",
        s=40,
        alpha=0.8,
        edgecolors="w",
        linewidth=0.8,
        label="Class 0",
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="gold",
        marker="^",
        s=40,
        alpha=0.9,
        edgecolors="k",
        linewidth=0.5,
        label="Class 1",
    )

    ax.set_title(title, fontsize=12)

axes[0].legend(loc="lower right")

fig.suptitle("MixedNB and GaussianNB on continuous data", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
