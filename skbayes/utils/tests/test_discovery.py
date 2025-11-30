# Authors: scikit-bayes developers
# License: BSD 3 clause

import pytest

from skbayes.utils.discovery import all_displays, all_estimators, all_functions


def test_all_estimators():
    estimators = all_estimators()
    # Debería ser 1 si solo tienes MixedNB
    assert len(estimators) == 1 
    assert estimators[0][0] == "MixedNB"

    # Verificar filtro de clasificadores
    estimators = all_estimators(type_filter="classifier")
    assert len(estimators) == 1


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    assert len(functions) == 3
