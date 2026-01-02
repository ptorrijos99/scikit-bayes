# Authors: scikit-bayes developers
# License: BSD 3 clause

import pytest

from skbayes.utils.discovery import all_displays, all_estimators, all_functions


def test_all_estimators():
    estimators = all_estimators()
    # Filter only estimators from skbayes modules (not sklearn imports)
    skbayes_estimators = [
        (name, cls) for name, cls in estimators
        if cls.__module__.startswith("skbayes")
    ]
    # Should be 5: MixedNB, AnDE, AnJE, ALR, WeightedAnDE
    assert len(skbayes_estimators) == 5
    estimator_names = [name for name, _ in skbayes_estimators]
    assert "MixedNB" in estimator_names
    assert "AnDE" in estimator_names
    assert "AnJE" in estimator_names
    assert "ALR" in estimator_names
    assert "WeightedAnDE" in estimator_names

    # Verify classifier filter returns skbayes classifiers
    classifiers = all_estimators(type_filter="classifier")
    skbayes_classifiers = [
        (name, cls) for name, cls in classifiers
        if cls.__module__.startswith("skbayes")
    ]
    assert len(skbayes_classifiers) == 5


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    assert len(functions) == 3
