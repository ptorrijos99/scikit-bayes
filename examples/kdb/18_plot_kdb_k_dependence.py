"""
=====================================================
Choosing k: KDB with Increasing Dependence
=====================================================

:class:`skbn.KDB` (K-Dependence Bayesian classifier) lets every attribute
depend on the class and on at most ``k`` other attributes. The attributes are
added to the network in decreasing order of their mutual information with the
class $I(A_i; C)$, and each one takes as parents the ``k`` attributes already
in the network with the highest conditional mutual information
$I(A_i; A_j \\mid C)$.

**The Scenario:**
Seven categorical attributes in two groups of three and a noise attribute.
In each group, $a$ is uniform, $b$ is a noisy copy of $a$, and
$s = a + b + y$ (with some noise). On its own, $a$ or $b$ says nothing about
the class; the class is recovered exactly only from the three attributes of a
group together.

**What Happens:**
The sums $s$ are the attributes most informative about the class, so they
are placed first. With ``k=2`` the last attribute of each group takes the
other two as parents and the model represents the whole group: it is the
most accurate. ``k=0`` is Naive Bayes, and ``k=1`` (a tree, comparable to
:class:`skbn.TAN`) only captures pairs. ``k=3`` can also represent the
groups, but its conditional probability tables have many more cells and need
more data: it loses with small training sets and catches up as they grow.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from skbn import KDB


# --- 1. Generate Data: Two Groups and a Noise Attribute ---
def make_data(n_samples, rng):
    y = rng.randint(0, 2, size=n_samples)
    columns = []
    for _ in range(2):
        a = rng.randint(0, 3, size=n_samples)
        b = np.where(rng.rand(n_samples) < 0.5, a, rng.randint(0, 3, size=n_samples))
        s = a + b + y
        s = np.where(rng.rand(n_samples) < 0.9, s, rng.randint(0, 6, size=n_samples))
        columns.extend([a, b, s])
    columns.append(rng.randint(0, 3, size=n_samples))  # noise
    return np.column_stack(columns), y


rng = np.random.RandomState(42)
X_train, y_train = make_data(5000, rng)
X_test, y_test = make_data(5000, rng)
names = ["a1", "b1", "s1", "a2", "b2", "s2", "noise"]

# --- 2. Learn the Network with k=2 ---
kdb = KDB(k=2).fit(X_train, y_train)
cmi = kdb.conditional_mutual_info_
print("Order:", [names[i] for i in kdb.feature_order_])
for i in kdb.feature_order_:
    print(f"  {names[i]:>5} <- {[names[j] for j in kdb.parents_[i]]}")

# --- 3. Accuracy as the Training Set Grows ---
sizes = [25, 50, 100, 200, 500, 1000, 2000, 5000]
styles = {
    0: ("goldenrod", "^"),
    1: ("teal", "o"),
    2: ("indigo", "s"),
    3: ("crimson", "D"),
}
accuracy = {k: [] for k in styles}
for n in sizes:
    for k in styles:
        model = KDB(k=k).fit(X_train[:n], y_train[:n])
        accuracy[k].append(accuracy_score(y_test, model.predict(X_test)))

for k in styles:
    print(f"k={k}: {np.round(accuracy[k], 3)}")

# --- 4. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
n_features = X_train.shape[1]
colors = ["indigo"] * 3 + ["crimson"] * 3 + ["gray"]

# (a) Mutual information with the class, in the order of placement
order = kdb.feature_order_
axes[0].bar(
    range(n_features), kdb.mutual_info_[order], color=[colors[i] for i in order]
)
axes[0].set_xticks(range(n_features))
axes[0].set_xticklabels([names[i] for i in order])
axes[0].set_ylabel("Nats")
axes[0].set_title("$I(A_i; C)$ in Order of Placement", fontsize=13)
axes[0].grid(axis="y", alpha=0.3)

# (b) The learned network (the class, parent of every attribute, is omitted)
angles = np.pi / 2 - 2 * np.pi * np.arange(n_features) / n_features
pos = np.column_stack([np.cos(angles), np.sin(angles)])
max_cmi = max(cmi[i, p] for i in range(n_features) for p in kdb.parents_[i])
for i in range(n_features):
    for p in kdb.parents_[i]:
        axes[1].annotate(
            "",
            xy=pos[i],
            xytext=pos[p],
            arrowprops=dict(
                arrowstyle="-|>",
                color="teal",
                lw=0.5 + 5 * cmi[i, p] / max_cmi,
                shrinkA=18,
                shrinkB=18,
            ),
        )
for i in range(n_features):
    axes[1].scatter(*pos[i], s=1100, color=colors[i], zorder=3)
    axes[1].text(
        *pos[i], names[i], color="white", fontsize=9, ha="center", va="center", zorder=4
    )
axes[1].set_xlim(-1.4, 1.4)
axes[1].set_ylim(-1.4, 1.4)
axes[1].set_aspect("equal")
axes[1].axis("off")
axes[1].set_title("Learned Network, k=2 (Width: $I(A_i; A_j \\mid C)$)")

# (c) Accuracy vs training size
for k, (color, marker) in styles.items():
    label = "k=0 (Naive Bayes)" if k == 0 else f"k={k}"
    axes[2].plot(sizes, accuracy[k], color=color, marker=marker, lw=2, label=label)
axes[2].set_xscale("log")
axes[2].set_xlabel("Training Instances")
axes[2].set_ylabel("Test Accuracy")
axes[2].set_title("Test Accuracy", fontsize=13)
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.suptitle("More Dependence Captures More, but Needs More Data", fontsize=16)
plt.show()
