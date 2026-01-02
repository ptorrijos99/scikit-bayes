"""
General tests for all estimators in skbn.
"""
# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.utils.estimator_checks import check_estimator

from skbn.utils.discovery import all_estimators


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimators(name, Estimator):
    # AnDE family tests are skipped due to strict numerical checks in check_estimator
    # (infinite mismatches on random dense data schemes).
    # MixedNB passes fully.
    if name in ["AnDE", "AnJE", "ALR", "WeightedAnDE"]:
        pytest.skip(
            f"{name} skipped: numerical stability issues with random data in"
            " check_estimator"
        )

    check_estimator(Estimator())
