"""
=====================================================
Gaussian children or discretized attributes
=====================================================

:class:`skbn.AnJE` and :class:`skbn.ALR` are defined on discrete attributes,
so they discretize every continuous attribute into ``n_bins`` bins.
:class:`skbn.GaussianAnJE` and :class:`skbn.GaussianALR` build the same
models on the factors of the SPnDEs: only the super-parents are discretized,
and every attribute keeps a Gaussian distribution as a child, as in HAODE.
``GaussianAnJE(n_dependence=1)`` is the counterpart of ``AnJE(n_join=2)``.

The data have eight continuous attributes whose distribution within each
class is Gaussian, and the models are trained on 500 samples with between 2
and 20 bins. Both ALR models use ``l2_reg=1``.

The discretized models depend on the number of bins: with few bins they lose
the position of the values within each bin, and with many bins each cell
holds few samples, which ALR overfits. In the Gaussian models only the
super-parents depend on the bins, so they change much less with ``n_bins``,
and they are better than the discretized models at every setting.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from skbn import ALR, AnJE, GaussianALR, GaussianAnJE

seeds = [0, 1, 2]
bins = [2, 3, 5, 10, 20]
n_train = 500

# Name, factory of n_bins, colour, marker
models = [
    ("AnJE", lambda b: AnJE(n_join=2, n_bins=b), "#009E73", "o"),
    ("ALR", lambda b: ALR(n_join=2, n_bins=b, l2_reg=1.0), "#D55E00", "^"),
    (
        "GaussianAnJE",
        lambda b: GaussianAnJE(n_dependence=1, n_bins=b),
        "#44AA99",
        "s",
    ),
    (
        "GaussianALR",
        lambda b: GaussianALR(n_dependence=1, n_bins=b, l2_reg=1.0),
        "#882255",
        "D",
    ),
]

shape = (len(seeds), len(bins), len(models))
res_ll, res_acc = np.zeros(shape), np.zeros(shape)
for i, seed in enumerate(seeds):
    X, y = make_classification(
        n_samples=n_train + 5000,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, random_state=seed
    )
    for j, n_bins in enumerate(bins):
        for k, (_, factory, _, _) in enumerate(models):
            proba = factory(n_bins).fit(X_train, y_train).predict_proba(X_test)
            res_ll[i, j, k] = log_loss(y_test, proba)
            res_acc[i, j, k] = accuracy_score(y_test, proba.argmax(axis=1))

mean_ll, mean_acc = res_ll.mean(axis=0), res_acc.mean(axis=0)
for k, (name, *_) in enumerate(models):
    print(f"{name:13s} log loss {np.round(mean_ll[:, k], 3)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, values, ylabel in zip(
    axes, [mean_ll, mean_acc], ["Test log loss", "Test accuracy"]
):
    for k, (name, _, color, marker) in enumerate(models):
        ax.plot(bins, values[:, k], marker=marker, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xticks(bins, [str(b) for b in bins])
    ax.minorticks_off()
    ax.set_xlabel("Number of bins")
    ax.set_ylabel(ylabel)
axes[0].set_yscale("log")
axes[0].set_yticks([0.2, 0.3, 0.5, 1, 2, 4], ["0.2", "0.3", "0.5", "1", "2", "4"])
axes[0].minorticks_off()
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
fig.suptitle(f"Discretized and Gaussian children, {n_train} training samples")
plt.show()
