"""Tests of the dispatch of the compiled kernels."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np
import pytest

from skbn._kernels import _Kernel, _pairwise_sum, _state_index, _threads


def _row_sums(x):
    out = np.zeros(x.shape[0])
    for i in nb.prange(x.shape[0]):
        out[i] = x[i].sum()
    return out


@pytest.mark.skipif(
    nb.config.DISABLE_JIT, reason="Without the JIT there is nothing to compile"
)
def test_kernel_compiles_the_parallel_version_only_for_several_threads():
    kernel = _Kernel(_row_sums)
    x = np.arange(12.0).reshape(4, 3)
    np.testing.assert_array_equal(kernel(x), x.sum(axis=1))
    assert kernel._parallel is None
    with _threads(None):
        kernel(x)
    assert kernel._parallel is None
    with _threads(2):
        np.testing.assert_array_equal(kernel(x), x.sum(axis=1))
    assert kernel._parallel is not None
    assert kernel._parallel.targetoptions.get("parallel")
    assert not kernel._serial.targetoptions.get("parallel")
    assert kernel._parallel.py_func.__qualname__ != kernel._serial.py_func.__qualname__


def test_threads_restores_the_previous_setting():
    previous = nb.get_num_threads()
    with _threads(2):
        with _threads(None):
            assert nb.get_num_threads() == 1
        assert nb.get_num_threads() == min(2, nb.config.NUMBA_NUM_THREADS)
    assert nb.get_num_threads() == previous


def test_state_index_of_large_grids_matches_unique():
    """Grids too large for a dense index are sorted by a radix sort, up to the
    largest key a 64-bit index allows."""
    rng = np.random.RandomState(0)
    for n_grid in (10**5, 2**40, 2**62):
        for high in (50, n_grid):
            keys = rng.randint(0, high, size=2000, dtype=np.int64)
            keys[:3] = n_grid - 1
            states, index = _state_index(keys, n_grid)
            np.testing.assert_array_equal(states, np.unique(keys))
            np.testing.assert_array_equal(index, np.searchsorted(states, keys))


@pytest.mark.parametrize("n", [1, 7, 8, 9, 40, 127, 128, 129, 300, 1000, 4099])
def test_pairwise_sum_has_the_bits_of_numpy_sum(n):
    # The compiled weighted objective relies on summing in NumPy's order
    rng = np.random.default_rng(n)
    x = rng.normal(size=(20, n)) * 10.0 ** rng.uniform(-3, 6, size=(20, n))
    for row in x:
        assert _pairwise_sum(row, 0, n) == 0.0 + row.sum()
