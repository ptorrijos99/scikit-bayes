"""
General tests for all estimators in skbn.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.utils.estimator_checks import check_estimator

from skbn import WeightedAnDE
from skbn.utils.discovery import all_estimators


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimators(name, Estimator):
    check_estimator(Estimator())


def test_modular_weighted_ande():
    # The modular hybrid is only identifiable at the class-specific granularities
    check_estimator(WeightedAnDE(modular=True, weight_level=3))
