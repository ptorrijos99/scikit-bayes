"""
=====================================================
MixedNB equivalence with BernoulliNB
=====================================================

When every feature is binary, :class:`skbn.MixedNB` detects them as
Bernoulli and fits the same model as :class:`sklearn.naive_bayes.BernoulliNB`.
The data are two binary features and a class that is 1 only when both
features are 1 (an AND gate). The plot shows the probability of class 1 on
the four possible feature combinations for each classifier; the two maps
coincide.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from sklearn.naive_bayes import BernoulliNB

from skbn import MixedNB

np.random.seed(42)
n_samples = 100
X = np.random.randint(0, 2, size=(n_samples, 2), dtype=int)
y = (X[:, 0] & X[:, 1]).astype(int)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X, y)
probs_bnb = bnb.predict_proba(X)

# MixedNB detects both features as 'bernoulli' (two unique values)
mnb = MixedNB(alpha=1.0)
mnb.fit(X, y)
probs_mnb = mnb.predict_proba(X)

try:
    assert_allclose(probs_bnb, probs_mnb, rtol=1e-7, atol=1e-7)
    equivalence_message = "Probabilities are identical."
except AssertionError as e:
    equivalence_message = f"Probabilities are NOT identical:\n{e}"

print(f"BernoulliNB vs MixedNB: {equivalence_message}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

models = [bnb, mnb]
titles = ["BernoulliNB (scikit-learn)", "MixedNB (skbn)"]

# Cell centers are used for prediction and cell edges for plotting
centers = np.array([0, 1])
edges = np.array([-0.5, 0.5, 1.5])

xx, yy = np.meshgrid(centers, centers)
grid_pred = np.c_[xx.ravel(), yy.ravel()]

for ax, model, title in zip(axes, models, titles):
    probs = model.predict_proba(grid_pred)[:, 1]
    Z = probs.reshape(xx.shape)

    ax.pcolormesh(
        edges,
        edges,
        Z,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="flat",
        alpha=0.8,
        edgecolors="none",
    )

    # Jitter the points because many of them overlap on 0/1
    x_jit = X[:, 0] + np.random.uniform(-0.15, 0.15, size=n_samples)
    y_jit = X[:, 1] + np.random.uniform(-0.15, 0.15, size=n_samples)

    ax.scatter(
        x_jit[y == 0],
        y_jit[y == 0],
        c="indigo",
        marker="o",
        s=50,
        alpha=0.8,
        edgecolors="w",
        linewidth=0.8,
        label="Class 0",
    )

    ax.scatter(
        x_jit[y == 1],
        y_jit[y == 1],
        c="gold",
        marker="^",
        s=50,
        alpha=0.9,
        edgecolors="k",
        linewidth=0.5,
        label="Class 1",
    )

    ax.set_title(title, fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.2, linestyle="--")

axes[0].set_ylabel("Feature 1 (Bernoulli)")
axes[0].set_xlabel("Feature 0 (Bernoulli)")
axes[1].set_xlabel("Feature 0 (Bernoulli)")
axes[0].legend(loc="lower left")

fig.suptitle("MixedNB and BernoulliNB on binary data", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
