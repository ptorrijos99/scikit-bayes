"""
====================================================================
Component or factor weights: data efficiency and cost
====================================================================

:class:`skbn.WeightedAnDE` learns discriminative weights on top of AnDE.
``weight_scope`` decides what a weight multiplies:

-   ``"component"``: a whole SPODE. At level 3 there is one weight per SPODE
    and class; at level 4, one per SPODE, class and value of its
    super-parent. There are few weights, and they only decide how much each
    SPODE is trusted.
-   ``"factor"``: each factor inside each SPODE, as in the weighted CCBN of
    Zaidi et al. (2017). Level 3 adds the value of the attribute and the
    class; level 4 also the value of the super-parent. These weights can
    discount an attribute that the SPODEs count several times.

The data have ten attributes, four of them redundant linear combinations of
the informative ones, and the models are trained on samples of increasing
size.

With 100 samples AnDE leads and every weighted model overfits. Between 500
and 2000 samples the component weights of level 3 match or beat AnDE, while
the factor weights still lag behind. With 10,000 samples the factor weights give the lowest log loss and
the highest accuracy, since the component weights cannot undo the
redundancy that every SPODE shares. The factor weights also take longer to
train than the component weights.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, StrMethodFormatter
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from skbn import AnDE, WeightedAnDE

seeds = [1, 2, 3]
sizes = [100, 500, 2000, 10000]

# Name, factory, colour, line style, marker
models_config = [
    ("AnDE", lambda: AnDE(n_dependence=1), "#0072B2", "-", "o"),
    (
        "Component weights, level 3",
        lambda: WeightedAnDE(weight_level=3, l2_reg=1e-3),
        "#CC79A7",
        "-",
        "s",
    ),
    (
        "Component weights, level 4",
        lambda: WeightedAnDE(weight_level=4, l2_reg=1e-3),
        "#CC79A7",
        "-",
        "D",
    ),
    (
        "Factor weights, level 3",
        lambda: WeightedAnDE(weight_scope="factor", weight_level=3, l2_reg=1e-3),
        "#CC79A7",
        "--",
        "^",
    ),
    (
        "Factor weights, level 4",
        lambda: WeightedAnDE(weight_scope="factor", weight_level=4, l2_reg=1e-3),
        "#CC79A7",
        "--",
        "v",
    ),
]

# Fit each model once so that the times below exclude the compilation of
# the kernels on their first use
X_small, y_small = make_classification(n_samples=200, n_features=10, random_state=0)
for _, factory, *_ in models_config:
    factory().fit(X_small, y_small)

# Results: [seed, size, model]
shape = (len(seeds), len(sizes), len(models_config))
res_ll, res_acc, res_time = np.zeros(shape), np.zeros(shape), np.zeros(shape)

for i, seed in enumerate(seeds):
    X, y = make_classification(
        n_samples=15000,
        n_features=10,
        n_informative=4,
        n_redundant=4,
        random_state=seed,
    )
    for j, n_train in enumerate(sizes):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n_train, test_size=5000, random_state=seed
        )
        for k, (_, factory, _, _, _) in enumerate(models_config):
            start = time.perf_counter()
            clf = factory().fit(X_train, y_train)
            res_time[i, j, k] = time.perf_counter() - start
            proba = clf.predict_proba(X_test)
            res_ll[i, j, k] = log_loss(y_test, proba)
            res_acc[i, j, k] = accuracy_score(y_test, proba.argmax(axis=1))
    print(f"Seed {seed} done.")

mean_ll, mean_acc, mean_time = (r.mean(axis=0) for r in (res_ll, res_acc, res_time))
for k, (name, *_) in enumerate(models_config):
    print(f"{name:28s} log loss {np.round(mean_ll[:, k], 3)}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
panels = [
    (mean_ll, "Test log loss"),
    (mean_acc, "Test accuracy"),
    (mean_time, "Training time (s)"),
]
for ax, (values, ylabel) in zip(axes, panels):
    for k, (name, _, color, style, marker) in enumerate(models_config):
        ax.plot(sizes, values[:, k], ls=style, marker=marker, color=color, label=name)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training set size")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
for ax in (axes[0], axes[2]):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
axes[0].set_yticks([0.2, 0.3, 0.5, 1, 2])
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5)

fig.suptitle("Component and factor weights in WeightedAnDE")
plt.show()
