"""
General tests for all estimators in skbn.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.utils.estimator_checks import check_estimator

from skbn import AnDE, WeightedAnDE
from skbn.utils.discovery import all_estimators


def _few_iterations(estimator):
    """``estimator`` with a few optimizer iterations, if it has any. These
    checks are about the scikit-learn API, not about how well the weights fit
    (tested on their own); with the kernels run as Python, as for coverage,
    the full optimization of every check would take tens of minutes."""
    if "max_iter" in estimator.get_params():
        estimator.set_params(max_iter=3)
    return estimator


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimators(name, Estimator):
    check_estimator(_few_iterations(Estimator()))


@pytest.mark.parametrize(
    "estimator",
    [
        AnDE(mean="geometric"),
        WeightedAnDE(mean="geometric"),
        WeightedAnDE(weight_scope="factor", weight_level=4),
        WeightedAnDE(mean="geometric", weight_scope="factor", weight_level=4),
        # The modular hybrid with arithmetic component weights is only
        # identifiable at the class-specific granularities
        WeightedAnDE(modular=True, weight_level=3),
        WeightedAnDE(modular=True, weight_scope="factor", weight_level=1),
    ],
    ids=repr,
)
def test_variants(estimator):
    check_estimator(_few_iterations(estimator))
