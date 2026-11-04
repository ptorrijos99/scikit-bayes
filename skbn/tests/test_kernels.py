"""Tests of the dispatch of the compiled kernels."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np

from skbn._kernels import _Kernel, _threads


def _row_sums(x):
    out = np.zeros(x.shape[0])
    for i in nb.prange(x.shape[0]):
        out[i] = x[i].sum()
    return out


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
