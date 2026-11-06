"""
=====================================================
XOR problem: AnDE and Naive Bayes
=====================================================

The XOR dataset has two continuous features. Class 0 lies in the first and
third quadrants (both features with the same sign) and class 1 in the second
and fourth. Each feature taken alone has the same distribution in both
classes, centered at 0, so Naive Bayes has nothing to learn from and stays
near chance level.

:class:`skbn.AnDE` with ``n_dependence=1`` conditions every feature on one
other feature, the super-parent, which it discretizes. Given the sign of the
super-parent, the other feature does separate the classes, and the ensemble
recovers the four quadrants. The figure shows the probability of class 1 for
:class:`skbn.MixedNB` and for AnDE, with the training accuracy in the titles.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

from skbn import AnDE, MixedNB

np.random.seed(42)
n_samples = 400

X = np.random.randn(n_samples, 2)

y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

mnb = MixedNB()
mnb.fit(X, y)

# The super-parent is discretized into 4 bins and the child is modeled as a
# Gaussian conditioned on the bin.
ande = AnDE(n_dependence=1, n_bins=4)
ande.fit(X, y)

models = [mnb, ande]
titles = ["MixedNB", "AnDE (n=1)"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, model, title in zip(axes, models, titles):
    acc = accuracy_score(y, model.predict(X))

    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict_proba",
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.8,
        ax=ax,
        vmin=0,
        vmax=1,
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

    ax.set_title(f"{title}\nAccuracy: {acc:.2f}", fontsize=12)
    ax.set_xlabel("Feature 1")

    ax.axvline(0, color="white", linestyle="--", alpha=0.3)
    ax.axhline(0, color="white", linestyle="--", alpha=0.3)

axes[0].set_ylabel("Feature 2")
axes[0].legend(loc="lower left")

fig.suptitle("XOR problem: probability of class 1", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
