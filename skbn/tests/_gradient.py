"""Check of analytic gradients against central differences."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose


def assert_gradient(fun, theta, grad, n_directions=3, eps=1e-5):
    """Assert that ``grad`` is the gradient of ``fun`` at ``theta``.

    Along random directions ``v`` the derivative ``grad @ v`` must match the
    central difference ``(fun(theta + eps v) - fun(theta - eps v)) / 2 eps``.
    A wrong entry of ``grad`` changes ``grad @ v`` along almost every
    direction, so a few directions check every entry with two evaluations
    each, where a difference per entry takes one evaluation per weight.
    """
    rng = np.random.RandomState(0)
    for _ in range(n_directions):
        v = rng.normal(size=theta.size)
        numeric = (fun(theta + eps * v) - fun(theta - eps * v)) / (2 * eps)
        assert_allclose(grad @ v, numeric, rtol=1e-6, atol=1e-6)
