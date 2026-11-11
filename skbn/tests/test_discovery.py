"""The common tests must see every estimator of the package."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.base import is_classifier

from skbn.tests._discovery import all_estimators


def test_all_estimators():
    estimators = dict(all_estimators())
    assert sorted(estimators) == [
        "ALR",
        "AnDE",
        "AnJE",
        "DecisionTreeDiscretizer",
        "GaussianALR",
        "GaussianAnJE",
        "KDB",
        "MDLPDiscretizer",
        "MixedNB",
        "TAN",
        "WeightedAnDE",
        "WeightedKDB",
        "WeightedTAN",
    ]
    classifiers = [name for name, cls in estimators.items() if is_classifier(cls())]
    assert len(classifiers) == 11  # all but the two discretizers
