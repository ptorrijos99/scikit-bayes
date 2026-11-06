"""
=====================================================
Cost of the order of dependence
=====================================================

An AnDE of order $n$ has one component for every set of $n$ attributes, so
an ensemble over $d$ attributes has $\\binom{d}{n}$ components:

-   $n = 0$: one component (Naive Bayes).
-   $n = 1$: $d$ components.
-   $n = 2$: $d(d-1)/2$ components.

Each component also has more parameters as the order grows, because its
tables are indexed by the values of $n$ super-parents.

:class:`skbn.AnDE` is fitted here at orders 0, 1 and 2 on datasets with a
growing number of attributes, and the figure shows the number of components
and the training time of each. The time of orders 0 and 1 barely changes
over this range, while that of order 2 grows with its number of components.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, StrMethodFormatter

from skbn import AnDE

rng = np.random.RandomState(42)
n_samples = 2000
n_features_grid = [5, 10, 15, 20, 30, 40]
orders = [0, 1, 2]
n_repeats = 3

X_full = rng.randint(0, 4, size=(n_samples, max(n_features_grid)))
y = rng.randint(0, 3, size=n_samples)

n_components = {n: [] for n in orders}
fit_times = {n: [] for n in orders}

for d in n_features_grid:
    X = X_full[:, :d]
    for n in orders:
        model = AnDE(n_dependence=n, categorical_features=list(range(d)))
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model.fit(X, y)
            times.append(time.perf_counter() - t0)
        n_components[n].append(len(model.ensemble_))
        fit_times[n].append(min(times))
    print(
        f"d = {d:2d}   components: {[n_components[n][-1] for n in orders]}   "
        f"fit time (s): {[round(fit_times[n][-1], 3) for n in orders]}"
    )

# Label, colour and marker of each order
styles = {
    0: ("$n = 0$ (Naive Bayes)", "#6E6E6E", "s"),
    1: ("$n = 1$ (A1DE)", "#0072B2", "o"),
    2: ("$n = 2$ (A2DE)", "#56B4E9", "^"),
}

fig, (ax_comp, ax_time) = plt.subplots(1, 2, figsize=(10, 4.2))

for n in orders:
    label, color, marker = styles[n]
    ax_comp.plot(
        n_features_grid, n_components[n], color=color, marker=marker, label=label
    )
    ax_time.plot(n_features_grid, fit_times[n], color=color, marker=marker)

for ax in (ax_comp, ax_time):
    ax.set_xlabel("Number of attributes")
    ax.set_xticks(n_features_grid)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
ax_comp.set_ylabel("Components in the ensemble")
ax_time.set_ylabel("Training time (s)")

fig.legend(*ax_comp.get_legend_handles_labels(), loc="outside lower center", ncol=3)
fig.suptitle(f"Cost of AnDE by order of dependence, {n_samples} training instances")
plt.show()
