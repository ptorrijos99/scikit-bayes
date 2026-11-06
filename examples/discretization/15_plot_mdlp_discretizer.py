"""
=====================================================
Supervised discretization: where the cuts fall
=====================================================

The figure compares where three discretizers place their cut-points on the
same continuous attribute. The attribute is skewed (most values are small)
and the class changes at two thresholds, $x = 2$ and $x = 6$, with some
label noise. The discretizers are:

1.  :class:`sklearn.preprocessing.KBinsDiscretizer` with equal frequency,
    which ignores the class, so its cuts follow the density of the attribute
    and pile up where the data are, not where the class changes.
2.  :class:`skbn.DecisionTreeDiscretizer`, which is supervised but produces
    as many bins as it is asked for.
3.  :class:`skbn.MDLPDiscretizer`, the method of Fayyad and Irani (1993). It
    splits recursively on class entropy and stops when the Minimum
    Description Length criterion says that a further cut is not worth its
    cost, so the data choose the number of bins.

MDLP recovers the two thresholds and nothing else.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

from skbn import DecisionTreeDiscretizer, MDLPDiscretizer

# A skewed attribute with two class thresholds
rng = np.random.RandomState(42)
n_samples = 1500
true_cuts = [2.0, 6.0]

x = rng.gamma(shape=2.0, scale=1.6, size=n_samples)
y = np.digitize(x, true_cuts)

# Label noise: 10% of the labels are replaced at random
noisy = rng.rand(n_samples) < 0.10
y[noisy] = rng.randint(0, 3, size=noisy.sum())
X = x.reshape(-1, 1)

kbins = KBinsDiscretizer(
    n_bins=5, encode="ordinal", strategy="quantile", subsample=None
).fit(X)
tree = DecisionTreeDiscretizer(n_bins=5, random_state=0).fit(X, y)
mdlp = MDLPDiscretizer().fit(X, y)

# Name, cut-points, colour
cuts = [
    ("KBinsDiscretizer, 5 bins", kbins.bin_edges_[0][1:-1], "#1A1A1A"),
    ("DecisionTreeDiscretizer, 5 bins", tree.thresholds_[0], "#999933"),
    ("MDLPDiscretizer", mdlp.cut_points_[0], "#661100"),
]

for name, c, _ in cuts:
    print(f"{name:32s} {np.round(np.sort(c), 2)}")

class_colors = ["#C4C4C4", "#8C8C8C", "#575757"]
bins = np.linspace(0, 14, 57)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, (name, c, color) in zip(axes, cuts):
    ax.hist(
        [x[y == k] for k in range(3)],
        bins=bins,
        stacked=True,
        color=class_colors,
        label=[f"Class {k}" for k in range(3)],
    )
    for t in true_cuts:
        ax.axvline(t, color="#1A1A1A", ls=":", lw=1.2)
    for t in c:
        ax.axvline(t, color=color, label=f"Cut-point, {name.split(',')[0]}")
    ax.set_title(f"{name}\n{len(c)} cut-points")
    ax.set_xlabel("Attribute value")
    ax.set_xlim(0, 14)
axes[0].set_ylabel("Count")

# One legend entry per class, per discretizer and for the true thresholds
handles, labels = [], []
for ax in axes:
    for h, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(h)
            labels.append(label)
(true_line,) = axes[0].plot([], [], color="#1A1A1A", ls=":", lw=1.2)
handles.append(true_line)
labels.append("True threshold")
# Classes on the first row of the legend, cut-points on the second
order = [0, 3, 1, 4, 2, 5, 6]
fig.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="outside lower center",
    ncol=4,
)

fig.suptitle("Cut-points of three discretizers on the same attribute")
plt.show()

# %%
# On a real dataset, iris, MDLP keeps two cut-points per attribute.
X_iris, y_iris = load_iris(return_X_y=True)
iris_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
iris_mdlp = MDLPDiscretizer().fit(X_iris, y_iris)
for name, c in zip(iris_names, iris_mdlp.cut_points_):
    print(f"{name:14s} {np.round(c, 2)}")
