"""
=====================================================
Three-dimensional XOR problem
=====================================================

The class is the parity of the signs of three features: class 1 when the
product of the signs is positive, which splits the cube into eight octants
alternating between the two classes. Each feature alone and each pair of
features is independent of the class.

The figure shows the probability of class 1 predicted by
:class:`skbn.MixedNB` (A0DE), :class:`skbn.AnDE` with ``n_dependence=1``
(A1DE) and with ``n_dependence=2`` (A2DE) on a regular grid of points, with a
sample of the training instances on top. Only the grid points where the
probability differs from 0.5 by more than 0.1 are drawn. A0DE and A1DE
cannot represent the parity and stay near chance level, whereas A2DE
conditions on pairs of features and recovers the octants.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import accuracy_score

from skbn import AnDE, MixedNB

np.random.seed(42)
n_samples = 2000
X = np.random.randn(n_samples, 3)
y = (np.sign(X[:, 0]) * np.sign(X[:, 1]) * np.sign(X[:, 2]) > 0).astype(int)

models = [MixedNB(), AnDE(n_dependence=1, n_bins=4), AnDE(n_dependence=2, n_bins=4)]
names = ["A0DE (MixedNB)", "A1DE (n=1)", "A2DE (n=2)"]

fitted_models = []
scores = []
for model in models:
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    fitted_models.append(model)
    scores.append(acc)

n_grid = 15
grid_range = np.linspace(-2.2, 2.2, n_grid)
xx, yy, zz = np.meshgrid(grid_range, grid_range, grid_range)
X_grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

fig = plt.figure(figsize=(18, 6))

for i, (model, name, score) in enumerate(zip(fitted_models, names, scores)):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")

    probs = model.predict_proba(X_grid)[:, 1]

    # Draw only the grid points where the model is not close to 0.5
    mask = np.abs(probs - 0.5) > 0.1

    xs = X_grid[mask, 0]
    ys = X_grid[mask, 1]
    zs = X_grid[mask, 2]
    c_vals = probs[mask]

    p = ax.scatter(
        xs,
        ys,
        zs,
        c=c_vals,
        cmap="viridis",
        s=30,
        alpha=0.3,
        edgecolors="none",
        marker="s",
    )

    # Subsample the training points to keep the 3D view readable
    mask_sub = np.random.choice(n_samples, 100, replace=False)
    X_sub = X[mask_sub]
    y_sub = y[mask_sub]

    ax.scatter(
        X_sub[y_sub == 0, 0],
        X_sub[y_sub == 0, 1],
        X_sub[y_sub == 0, 2],
        c="indigo",
        marker="o",
        s=20,
        alpha=0.9,
        edgecolors="w",
        label="Class 0",
    )

    ax.scatter(
        X_sub[y_sub == 1, 0],
        X_sub[y_sub == 1, 1],
        X_sub[y_sub == 1, 2],
        c="gold",
        marker="^",
        s=20,
        alpha=1.0,
        edgecolors="k",
        label="Class 1",
    )

    ax.plot([-2.5, 2.5], [0, 0], [0, 0], "k-", lw=1, alpha=0.3)
    ax.plot([0, 0], [-2.5, 2.5], [0, 0], "k-", lw=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-2.5, 2.5], "k-", lw=1, alpha=0.3)

    ax.set_title(f"{name}\nAccuracy: {score:.3f}", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)

cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
fig.colorbar(p, cax=cbar_ax, label="Probability of class 1")

axes_3d = fig.get_axes()[0]
axes_3d.legend(loc="lower left", fontsize=9)

plt.suptitle("Three-dimensional XOR problem: probability of class 1", fontsize=16)
plt.show()
