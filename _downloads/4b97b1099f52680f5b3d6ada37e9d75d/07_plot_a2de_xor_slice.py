"""
=====================================================
Slice of the three-dimensional XOR problem
=====================================================

In the three-dimensional XOR problem the class is the parity of the signs of
three features: class 1 when the product of the signs is positive. Fixing one
feature turns it into a two-dimensional problem. At z = 1 the class is 1 when
x and y have the same sign, a checkerboard of four quadrants in the (x, y)
plane.

The figure shows the probability of class 1 on that slice for
:class:`skbn.MixedNB` (A0DE), :class:`skbn.AnDE` with ``n_dependence=1``
(A1DE) and with ``n_dependence=2`` (A2DE). The dashed line is the 0.5
boundary and the points are the training instances with z between 0.5 and
1.5. The titles give the accuracy over the whole training set, not only the
slice. A0DE and A1DE look at one feature at a time, and the parity of three
signs is invisible to them, so they stay near chance level. A2DE conditions
on pairs of features and draws the checkerboard.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from skbn import AnDE, MixedNB

np.random.seed(42)
n_samples = 2000
X = np.random.randn(n_samples, 3)
y = (np.sign(X[:, 0]) * np.sign(X[:, 1]) * np.sign(X[:, 2]) > 0).astype(int)

models = [MixedNB(), AnDE(n_dependence=1, n_bins=4), AnDE(n_dependence=2, n_bins=4)]
names = ["MixedNB (A0DE)", "AnDE, $n = 1$ (A1DE)", "AnDE, $n = 2$ (A2DE)"]

for model in models:
    model.fit(X, y)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)

# Grid on the slice z = 1
h = 0.05
limit = 3
xx, yy = np.meshgrid(np.arange(-limit, limit, h), np.arange(-limit, limit, h))
grid_flat = np.c_[xx.ravel(), yy.ravel()]
X_test_slice = np.hstack([grid_flat, np.ones((len(grid_flat), 1))])

for ax, model, name in zip(axes, models, names):
    acc = accuracy_score(y, model.predict(X))

    probs = model.predict_proba(X_test_slice)[:, 1]
    Z_probs = probs.reshape(xx.shape)

    mesh = ax.pcolormesh(
        xx, yy, Z_probs, cmap="viridis", vmin=0, vmax=1, shading="auto", alpha=0.8
    )
    ax.contour(xx, yy, Z_probs, levels=[0.5], colors="black", linestyles="--")

    # Training points with z in [0.5, 1.5] stand for the slice
    mask_slice = (X[:, 2] > 0.5) & (X[:, 2] < 1.5)

    ax.scatter(
        X[mask_slice & (y == 0), 0],
        X[mask_slice & (y == 0), 1],
        c="indigo",
        marker="o",
        s=20,
        alpha=0.9,
        edgecolors="w",
        linewidth=0.6,
        label="Class 0",
    )
    ax.scatter(
        X[mask_slice & (y == 1), 0],
        X[mask_slice & (y == 1), 1],
        c="gold",
        marker="^",
        s=20,
        alpha=0.9,
        edgecolors="k",
        linewidth=0.4,
        label="Class 1",
    )

    ax.set_title(f"{name}\nAccuracy on all the data: {acc:.3f}")
    ax.set_xlabel("Feature $x$")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_box_aspect(1)

axes[0].set_ylabel("Feature $y$")
fig.colorbar(mesh, ax=axes, label="Probability of class 1")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
fig.suptitle("Slice $z = 1$ of the three-dimensional XOR problem")
plt.show()
