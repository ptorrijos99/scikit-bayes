"""
=====================================================
MixedNB on single-type data
=====================================================

On a table whose features all have one type, :class:`skbn.MixedNB` fits the
same model as the Naive Bayes estimator of scikit-learn for that type:
:class:`~sklearn.naive_bayes.GaussianNB` for continuous features,
:class:`~sklearn.naive_bayes.BernoulliNB` for binary ones and
:class:`~sklearn.naive_bayes.CategoricalNB` for discrete ones. MixedNB infers
the type of each feature from its values, so no type is declared here.

Each panel compares the probabilities of class 1 that the two estimators
give on held-out samples of one dataset: every point lies on the diagonal,
and the largest difference, in the title, is at the level of rounding.
"""

# Author: The scikit-bayes Developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB

from skbn import MixedNB

rng = np.random.RandomState(0)
n = 1000

X_gauss, y_gauss = make_classification(
    n_samples=n, n_features=4, n_informative=3, n_redundant=0, random_state=0
)

# Binary features whose rate of ones depends on the class
y_bin = rng.randint(2, size=n)
X_bin = (rng.rand(n, 6) < np.where(y_bin[:, None] == 1, 0.65, 0.35)).astype(int)

# Categorical features with 3 to 6 values, shifted by the class
y_cat = rng.randint(2, size=n)
cards = [3, 4, 5, 6]
X_cat = np.column_stack(
    [(rng.randint(c, size=n) + y_cat * rng.binomial(1, 0.6, n)) % c for c in cards]
)

datasets = [
    ("Continuous features", X_gauss, y_gauss, GaussianNB()),
    ("Binary features", X_bin, y_bin, BernoulliNB()),
    ("Categorical features", X_cat, y_cat, CategoricalNB(min_categories=cards)),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
for ax, (title, X, y, sk_model) in zip(axes, datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    p_sk = sk_model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    p_mixed = MixedNB().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    diff = np.abs(p_sk - p_mixed).max()
    print(f"{title}: largest difference {diff:.1e}")

    ax.plot([0, 1], [0, 1], color="#1A1A1A", lw=1, ls="--")
    ax.scatter(p_sk, p_mixed, s=16, alpha=0.6, color="#6E6E6E")
    ax.set_title(f"{title}\nLargest difference: {diff:.1g}")
    ax.set_xlabel(f"P(class 1), {type(sk_model).__name__}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
axes[0].set_ylabel("P(class 1), MixedNB")

fig.suptitle("MixedNB and the Naive Bayes estimators of scikit-learn")
plt.show()
