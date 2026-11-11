"""The public import paths of the estimators."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module

import pytest

import skbn


@pytest.mark.parametrize("name", skbn.__all__)
def test_every_public_name_imports_from_the_package(name):
    namespace = {}
    exec(f"from skbn import {name}", namespace)
    assert namespace[name] is getattr(skbn, name)


@pytest.mark.parametrize(
    "module, name",
    [
        ("skbn.ande", "AnDE"),
        ("skbn.ande", "WeightedAnDE"),
        ("skbn.anje", "AnJE"),
        ("skbn.anje", "ALR"),
        ("skbn.anje", "GaussianAnJE"),
        ("skbn.anje", "GaussianALR"),
        ("skbn.tan", "TAN"),
        ("skbn.tan", "WeightedTAN"),
        ("skbn.kdb", "KDB"),
        ("skbn.kdb", "WeightedKDB"),
        ("skbn.mixed_nb", "MixedNB"),
        ("skbn.discretization", "MDLPDiscretizer"),
        ("skbn.discretization", "DecisionTreeDiscretizer"),
    ],
)
def test_every_estimator_imports_from_its_module(module, name):
    estimator = getattr(import_module(module), name)
    assert estimator is getattr(skbn, name)
    assert estimator.__module__ == module
