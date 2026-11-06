"""
=====================================================
MixedNB equivalence with CategoricalNB
=====================================================

When every feature is discrete, :class:`skbn.MixedNB` detects them as
categorical and fits the same model as
:class:`sklearn.naive_bayes.CategoricalNB`. The data are two categorical
features, with 3 and 4 categories, and a class that is 1 when the sum of the
two category codes is at least 3. The plot shows the probability of class 1
for every combination of categories for each classifier; the two maps
coincide.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from sklearn.naive_bayes import CategoricalNB

from skbn import MixedNB

np.random.seed(42)
n_samples = 150
X = np.zeros((n_samples, 2), dtype=int)
X[:, 0] = np.random.randint(0, 3, size=n_samples)
X[:, 1] = np.random.randint(0, 4, size=n_samples)
y = (X[:, 0] + X[:, 1] >= 3).astype(int)

cnb = CategoricalNB(alpha=1.0)
cnb.fit(X, y)
probs_cnb = cnb.predict_proba(X)

# MixedNB detects both features as 'categorical'
mnb = MixedNB(alpha=1.0)
mnb.fit(X, y)
probs_mnb = mnb.predict_proba(X)

try:
    assert_allclose(probs_cnb, probs_mnb, rtol=1e-7, atol=1e-7)
    equivalence_message = "Probabilities are identical."
except AssertionError as e:
    equivalence_message = f"Probabilities are NOT identical:\n{e}"

print(f"CategoricalNB vs MixedNB: {equivalence_message}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

models = [cnb, mnb]
titles = ["CategoricalNB (scikit-learn)", "MixedNB (skbn)"]

# Cell centers are used for prediction and cell edges for plotting
x_centers = np.arange(3)
x_edges = np.arange(4) - 0.5

y_centers = np.arange(4)
y_edges = np.arange(5) - 0.5

xx, yy = np.meshgrid(x_centers, y_centers)
grid_pred = np.c_[xx.ravel(), yy.ravel()]

for ax, model, title in zip(axes, models, titles):
    probs = model.predict_proba(grid_pred)[:, 1]
    Z = probs.reshape(xx.shape)

    ax.pcolormesh(
        x_edges,
        y_edges,
        Z,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="flat",
        alpha=0.8,
        edgecolors="none",
    )

    # Jitter the points because many of them overlap
    x_jit = X[:, 0] + np.random.uniform(-0.2, 0.2, size=n_samples)
    y_jit = X[:, 1] + np.random.uniform(-0.2, 0.2, size=n_samples)

    ax.scatter(
        x_jit[y == 0],
        y_jit[y == 0],
        c="indigo",
        marker="o",
        s=40,
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
        s=40,
        alpha=0.9,
        edgecolors="k",
        linewidth=0.5,
        label="Class 1",
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature 0 (Categorical)")
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.grid(True, alpha=0.2, linestyle="--")

axes[0].set_ylabel("Feature 1 (Categorical)")
axes[0].legend(loc="lower right")

fig.suptitle("MixedNB and CategoricalNB on discrete data", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
