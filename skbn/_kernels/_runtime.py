"""Dispatch of the compiled kernels, and what all of them share.

:func:`_threads` sets the number of threads of the kernels called from a
Python thread, and :class:`_Kernel` compiles the parallel version of a kernel
only once it runs with more than one thread. The attribute kinds, the
clamped log zero, the block of samples evaluated together and the inlined
scalar helpers (clamping, binary search, Gaussian log-density) are used by
the kernels of every family.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import threading
import types
from contextlib import contextmanager

import numba as nb
import numpy as np
from joblib import effective_n_jobs

IGNORED, DISCRETE, GAUSSIAN = 0, 1, 2

# Value that np.nan_to_num(nan=-1e10, neginf=-1e10) gives a degenerate
# log-likelihood (zero probability or variance)
LOG_ZERO = -1e10
_FLOAT_MAX = np.finfo(np.float64).max


# Samples evaluated together, component by component, so that the tables of
# a component stay in cache while the block is evaluated
_BLOCK_ROWS = 256


# Whether the kernels called from this Python thread run their parallel
# version (set by _threads)
_state = threading.local()


@contextmanager
def _threads(n_jobs):
    """Run the kernels with the number of threads that ``n_jobs`` asks for
    (``None`` is 1, negative values count back from the number of CPUs)."""
    n = max(1, min(effective_n_jobs(n_jobs), nb.config.NUMBA_NUM_THREADS))
    previous = nb.get_num_threads(), getattr(_state, "parallel", False)
    nb.set_num_threads(n)
    _state.parallel = n > 1
    try:
        yield
    finally:
        nb.set_num_threads(previous[0])
        _state.parallel = previous[1]


class _Kernel:
    """A kernel compiled serially, and with ``parallel=True`` only the first
    time it runs with more than one thread: the parallel version takes two to
    five times longer to compile, a cost single-threaded use does not pay.
    Both versions are cached on disk."""

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self._func = func
        self._serial = nb.njit(cache=True)(func)
        self._parallel = None

    def __call__(self, *args):
        if not getattr(_state, "parallel", False):
            return self._serial(*args)
        if self._parallel is None:
            func = self._func
            # A copy under another name, so that the cache keeps both versions
            twin = types.FunctionType(
                func.__code__, func.__globals__, func.__name__ + "_parallel"
            )
            twin.__qualname__ = func.__qualname__ + "_parallel"
            self._parallel = nb.njit(parallel=True, cache=True)(twin)
        return self._parallel(*args)


@nb.njit(cache=True, inline="always")
def _clamp(v):
    """``np.nan_to_num(v, nan=-1e10, neginf=-1e10)``."""
    if v != v or v == -np.inf:
        return LOG_ZERO
    if v == np.inf:
        return _FLOAT_MAX
    return v


@nb.njit(cache=True, inline="always")
def _find(keys, lo, hi, key):
    """Position of ``key`` in the sorted ``keys[lo:hi]``, or -1."""
    end = hi
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and keys[lo] == key:
        return lo
    return -1


@nb.njit(cache=True, inline="always")
def _gaussian_ll(x, mean, var, logn):
    """Gaussian log-density, clamped as ``np.nan_to_num``; ``logn`` is
    ``log(2 pi var)``, from :func:`gaussian_log_norms`."""
    diff = x - mean
    return _clamp(-0.5 * (diff * diff / var + logn))


@nb.njit(cache=True)
def gaussian_log_norms(var):
    """``log(2 pi var)`` of every Gaussian: the term of the log-density that
    does not depend on the sample, computed once per table instead of once
    per evaluation."""
    out = np.empty(var.shape)
    for s in range(var.shape[0]):
        for g in range(var.shape[1]):
            out[s, g] = np.log(2.0 * np.pi * var[s, g])
    return out
