"""
=====================================================
The Cost of the Order of Dependence
=====================================================

An AnDE of order $n$ has one component for every set of $n$ attributes, so
an ensemble over $d$ attributes has $\\binom{d}{n}$ components:

-   $n = 0$: one component (Naive Bayes).
-   $n = 1$: $d$ components.
-   $n = 2$: $d(d-1)/2$ components.

Each component also has more parameters as the order grows, because its
tables are indexed by the values of $n$ super-parents.

This example fits :class:`skbn.AnDE` at orders 0, 1 and 2 on datasets with
a growing number of attributes and measures the number of components and
the training time. Order 2 is affordable with a few tens of attributes and
becomes expensive well before a few hundred.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np

from skbn import AnDE

# --- 1. Setup ---
rng = np.random.RandomState(42)
n_samples = 2000
n_features_grid = [5, 10, 15, 20, 30, 40]
orders = [0, 1, 2]
n_repeats = 3

X_full = rng.randint(0, 4, size=(n_samples, max(n_features_grid)))
y = rng.randint(0, 3, size=n_samples)

# --- 2. Measure ---
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

# --- 3. Plotting ---
styles = {0: ("gray", "s"), 1: ("teal", "o"), 2: ("indigo", "^")}

fig, (ax_comp, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))

for n in orders:
    color, marker = styles[n]
    ax_comp.plot(
        n_features_grid,
        n_components[n],
        color=color,
        marker=marker,
        linewidth=2,
        label=f"n = {n}",
    )
    ax_time.plot(
        n_features_grid,
        fit_times[n],
        color=color,
        marker=marker,
        linewidth=2,
        label=f"n = {n}",
    )

ax_comp.set_yscale("log")
ax_comp.set_xlabel("Number of Attributes")
ax_comp.set_ylabel("Components in the Ensemble")
ax_comp.set_title("Size of the Ensemble", fontsize=13)
ax_comp.legend(title="Order")
ax_comp.grid(alpha=0.3, which="both")

ax_time.set_yscale("log")
ax_time.set_xlabel("Number of Attributes")
ax_time.set_ylabel("Training Time (s)")
ax_time.set_title(f"Training Time ({n_samples} Instances)", fontsize=13)
ax_time.grid(alpha=0.3, which="both")

fig.suptitle("How the Cost of AnDE Grows with Its Order", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.show()
