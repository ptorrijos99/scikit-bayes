"""
=====================================================
Arithmetic or Geometric Mean: A Sensor That Fails
=====================================================

:class:`skbn.AnDE` and :class:`skbn.AnJE` fit exactly the same components.
They differ only in how they combine them:

-   **AnDE** averages the joint probabilities of the components
    (arithmetic mean), so the component that explains an instance best
    dominates its prediction.
-   **AnJE** averages their logarithms (geometric mean), so every component
    counts the same, and a few components that give a class a probability
    close to zero are enough to rule that class out.

On clean data the two rules give almost the same predictions. They
separate when part of the evidence is wrong.

**The Scenario:**
Five ordinary attributes and one very precise sensor. In training the
sensor is always right. At test time a fraction of its readings are
faulty and point to the opposite class, while the other five attributes
still point to the true one.

**What Happens:**
The components that have the sensor as a child give the true class a
negligible probability, because the reading is many standard deviations
away from what they expect. The component that has the sensor as its
super-parent does not: it models the other attributes given the reading,
and they still fit the true class. AnDE follows that component on about
half of the faulty instances. AnJE averages it with the other five and is
confidently wrong on most of them, which costs it accuracy and, above all,
log loss.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from skbn import AnDE, AnJE

# --- 1. Generate Data: Five Ordinary Attributes and a Precise Sensor ---
SENSOR_SHIFT, SENSOR_STD = 5.0, 0.8


def make_data(n_samples, rng):
    y = rng.randint(0, 2, size=n_samples)
    ordinary = rng.normal(loc=2.0 * y[:, None], scale=1.0, size=(n_samples, 5))
    sensor = rng.normal(loc=SENSOR_SHIFT * y, scale=SENSOR_STD)
    return np.column_stack([ordinary, sensor]), y


def add_faults(X, y, fraction, rng):
    """Make a fraction of the sensor readings point to the opposite class."""
    X = X.copy()
    faulty = rng.rand(len(X)) < fraction
    X[faulty, -1] = rng.normal(
        loc=SENSOR_SHIFT * (1 - y[faulty]), scale=SENSOR_STD, size=faulty.sum()
    )
    return X, faulty


rng = np.random.RandomState(42)
X_train, y_train = make_data(2000, rng)
X_test, y_test = make_data(20000, rng)

# --- 2. Fit the Same Components with Both Aggregation Rules ---
models = {
    "AnDE (Arithmetic Mean)": (AnDE(n_dependence=1), "teal", "o"),
    "AnJE (Geometric Mean)": (AnJE(n_dependence=1), "goldenrod", "^"),
}
for model, _, _ in models.values():
    model.fit(X_train, y_train)

# --- 3. Evaluate as the Sensor Fails More Often ---
fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]
res_ll = {name: [] for name in models}
res_acc = {name: [] for name in models}
p_true_faulty = {}

for fraction in fractions:
    X_faulty, faulty = add_faults(X_test, y_test, fraction, np.random.RandomState(0))
    for name, (model, _, _) in models.items():
        proba = model.predict_proba(X_faulty)
        res_ll[name].append(log_loss(y_test, proba))
        res_acc[name].append(accuracy_score(y_test, proba.argmax(axis=1)))
        if fraction == 0.05:
            # Probability given to the true class on the faulty readings
            p_true_faulty[name] = proba[faulty, y_test[faulty]]

for name in models:
    print(name)
    print("  Log loss:", np.round(res_ll[name], 3))
    print("  Accuracy:", np.round(res_acc[name], 3))
    print(
        "  Correct on faulty readings (5% faulty):"
        f" {np.mean(p_true_faulty[name] > 0.5):.2f}"
    )

# --- 4. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
percent = [100 * f for f in fractions]

for name, (_, color, marker) in models.items():
    axes[0].plot(
        percent, res_ll[name], color=color, marker=marker, linewidth=2, label=name
    )
    axes[1].plot(
        percent, res_acc[name], color=color, marker=marker, linewidth=2, label=name
    )
    axes[2].hist(
        np.log10(np.clip(p_true_faulty[name], 1e-12, 1)),
        bins=np.linspace(-12, 0, 37),
        color=color,
        alpha=0.6,
        label=name,
    )

axes[0].set_title("Test Log Loss", fontsize=13)
axes[0].set_ylabel("Log Loss (Lower Is Better)")
axes[1].set_title("Test Accuracy", fontsize=13)
axes[1].set_ylabel("Accuracy")
for ax in axes[:2]:
    ax.set_xlabel("Faulty Sensor Readings (%)")
    ax.grid(alpha=0.3)
    ax.legend()

axes[2].axvline(np.log10(0.5), color="gray", linestyle=":", linewidth=1.5)
axes[2].set_title("Faulty Readings Only (5% Faulty)", fontsize=13)
axes[2].set_xlabel("Probability Given to the True Class")
axes[2].set_ylabel("Test Instances")
axes[2].set_xticks(range(-12, 1, 2))
axes[2].set_xticklabels([f"$10^{{{k}}}$" for k in range(-12, 1, 2)])
axes[2].grid(alpha=0.3)
axes[2].legend(loc="upper left")

fig.suptitle("Same Components, Two Ways of Combining Them", fontsize=16)
plt.show()
