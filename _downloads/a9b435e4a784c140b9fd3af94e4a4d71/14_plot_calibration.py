"""
=====================================================
Calibration: Generative vs Hybrid Models
=====================================================

A classifier is calibrated when, among the instances to which it gives a
probability of 0.8, about 80% belong to the predicted class. This example
draws the reliability diagrams of five models of the library.

**The Scenario:**
A binary problem with a few informative attributes and many redundant
copies of them. Redundant attributes are counted several times by a model
that assumes independence, which pushes its probabilities towards 0 and 1.

**The Models:**

1.  :class:`skbn.MixedNB`: Naive Bayes, the most affected.
2.  :class:`skbn.AnDE`: conditioning on a super-parent absorbs part of the
    redundancy.
3.  :class:`skbn.WeightedAnDE` with component weights: one weight per SPODE
    and class. It decides how much each SPODE is trusted, but every SPODE
    still counts the redundant children several times.
4.  :class:`skbn.WeightedAnDE` with factor weights: one weight per factor,
    value and class inside each SPODE, which can discount each redundant
    child.
5.  :class:`skbn.ALR`: one weight per pair of attributes over the pairwise
    joins of the AnJE, which also discounts redundant pairs.

A curve below the diagonal on the right and above it on the left means
overconfidence. The histograms show how often each model predicts each
probability.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from skbn import ALR, AnDE, MixedNB, WeightedAnDE

# --- 1. Generate Data with Redundant Attributes ---
X, y = make_classification(
    n_samples=12000,
    n_features=14,
    n_informative=4,
    n_redundant=10,
    n_clusters_per_class=2,
    class_sep=0.8,
    flip_y=0.05,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# --- 2. Fit Models ---
models = {
    "Naive Bayes": (MixedNB(), "gray", "s"),
    "AnDE": (AnDE(n_dependence=1), "teal", "o"),
    "Component Weights": (WeightedAnDE(weight_level=3), "sandybrown", "^"),
    "Factor Weights": (
        WeightedAnDE(weight_scope="factor", weight_level=3),
        "mediumseagreen",
        "v",
    ),
    "ALR (Pairs)": (ALR(n_join=2, weight_level=1), "indigo", "D"),
}

probas = {}
for name, (model, _, _) in models.items():
    model.fit(X_train, y_train)
    probas[name] = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:18s} log loss {log_loss(y_test, probas[name]):.3f}   "
        f"Brier {brier_score_loss(y_test, probas[name]):.3f}"
    )

# --- 3. Plotting ---
fig, (ax_rel, ax_hist) = plt.subplots(
    2, 1, figsize=(9, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
)

ax_rel.plot([0, 1], [0, 1], color="black", linestyle="--", label="Perfect Calibration")
bins = np.linspace(0, 1, 21)

for name, (_, color, marker) in models.items():
    frac_pos, mean_pred = calibration_curve(
        y_test, probas[name], n_bins=12, strategy="quantile"
    )
    ll = log_loss(y_test, probas[name])
    ax_rel.plot(
        mean_pred,
        frac_pos,
        color=color,
        marker=marker,
        markersize=6,
        linewidth=2,
        label=f"{name} (Log Loss {ll:.3f})",
    )
    ax_hist.hist(
        probas[name], bins=bins, histtype="step", color=color, linewidth=2, label=name
    )

ax_rel.set_ylabel("Observed Fraction of Class 1")
ax_rel.set_title("Reliability Diagram", fontsize=13)
ax_rel.legend(loc="upper left")
ax_rel.grid(alpha=0.3)

ax_hist.set_xlabel("Predicted Probability of Class 1")
ax_hist.set_ylabel("Test Instances")
ax_hist.set_yscale("log")
ax_hist.set_title("Distribution of the Predicted Probabilities", fontsize=13)
ax_hist.grid(alpha=0.3)

fig.suptitle("Redundant Attributes Make Generative Models Overconfident", fontsize=15)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
