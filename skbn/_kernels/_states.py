"""Augmented states of the training rows, shared by the SPnDEs and the
conditionals of TAN and KDB.

An augmented state is a value of ``Y* = (y, x_p)``, the class and the
parents of a factor. These helpers key the training rows by their state,
index the states seen in training and count the values of a child in each
state; :class:`EvaluationInputs` holds the samples to evaluate.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import numba as nb
import numpy as np

# Augmented-state grids up to this many cells per training row are indexed
# with a dense array; larger ones are sorted.
_DENSE_CELLS_PER_ROW = 4


@nb.njit(cache=True)
def _augmented_keys(y, pcodes_t, parents, strides, n_combos):
    """Key ``y * n_combos + parent_key`` of every training row, from the
    parent codes transposed, shape (n_features, n): the parent columns are
    read whole, instead of a row of every attribute per training row."""
    n = y.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for r in range(n):
        keys[r] = y[r] * n_combos
    for q in range(parents.shape[0]):
        column, stride = pcodes_t[parents[q]], strides[q]
        for r in range(n):
            keys[r] += column[r] * stride
    return keys


@nb.njit(cache=True)
def _state_index(keys, n_grid):
    """Sorted distinct values of ``keys`` (all in ``[0, n_grid)``) and the
    position of every key among them."""
    n = keys.shape[0]
    if n_grid <= _DENSE_CELLS_PER_ROW * n + 1024:
        pos = np.full(n_grid, -1, dtype=np.int64)
        for r in range(n):
            pos[keys[r]] = 0
        n_states = 0
        for g in range(n_grid):
            if pos[g] == 0:
                pos[g] = n_states
                n_states += 1
        states = np.empty(n_states, dtype=np.int64)
        for g in range(n_grid):
            if pos[g] >= 0:
                states[pos[g]] = g
        index = np.empty(n, dtype=np.int64)
        for r in range(n):
            index[r] = pos[keys[r]]
        return states, index
    # Larger grids: order the rows by key with a radix sort, 16 bits a pass
    # (np.unique would compile NumPy's whole sorting machinery)
    order = np.arange(n)
    spare = np.empty(n, dtype=np.int64)
    count = np.empty((1 << 16) + 1, dtype=np.int64)
    shift = 0
    while shift < 64 and (n_grid - 1) >> shift > 0:
        count[:] = 0
        for r in range(n):
            count[((keys[order[r]] >> shift) & 0xFFFF) + 1] += 1
        for b in range(1 << 16):
            count[b + 1] += count[b]
        for r in range(n):
            digit = (keys[order[r]] >> shift) & 0xFFFF
            spare[count[digit]] = order[r]
            count[digit] += 1
        order, spare = spare, order
        shift += 16
    states = np.empty(n, dtype=np.int64)
    index = np.empty(n, dtype=np.int64)
    n_states = 0
    for t in range(n):
        r = order[t]
        if t == 0 or keys[r] != keys[order[t - 1]]:
            states[n_states] = keys[r]
            n_states += 1
        index[r] = n_states - 1
    return states[:n_states].copy(), index


@nb.njit(cache=True)
def _value_cells(index, codes, j, card, n_states):
    """Cells ``index[r] * card + codes[r, j]`` (local state, value of
    attribute ``j``) of the training rows, sorted and distinct, and the
    position of every row among them."""
    n = index.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for r in range(n):
        keys[r] = index[r] * card + codes[r, j]
    return _state_index(keys, n_states * card)


# Tables of a pair or triple up to this many cells are counted in four
# copies (see _histogram)
_SMALL_TABLE = 1024


@nb.njit(cache=True, inline="always")
def _histogram(base, card, x, counts, size):
    """``counts[base[r] * card + x[r]] += 1`` over every sample ``r``, in the
    first ``size`` entries of ``counts``. The cells are counted in an array
    of the function, which cannot alias the inputs: counted in ``counts``,
    the loop reloads them after every increment. A small table is counted in
    four copies, the samples taken in turns, so that consecutive samples in
    the same cell do not wait on each other's increment; the counts are the
    same. Inlined: called, the loop runs 1.5 times slower."""
    n = base.shape[0]
    if size > _SMALL_TABLE:
        local = np.zeros(size, dtype=np.int64)
        for r in range(n):
            local[base[r] * card + x[r]] += 1
        for k in range(size):
            counts[k] = local[k]
        return
    copies = np.zeros(4 * size, dtype=np.int64)
    end = n - n % 4
    for r in range(0, end, 4):
        copies[base[r] * card + x[r]] += 1
        copies[size + base[r + 1] * card + x[r + 1]] += 1
        copies[2 * size + base[r + 2] * card + x[r + 2]] += 1
        copies[3 * size + base[r + 3] * card + x[r + 3]] += 1
    for r in range(end, n):
        copies[base[r] * card + x[r]] += 1
    for k in range(size):
        counts[k] = copies[k] + copies[size + k] + copies[2 * size + k]
        counts[k] += copies[3 * size + k]


class EvaluationInputs(NamedTuple):
    """Per-sample inputs of the evaluation kernels (``inputs`` of
    :func:`component_scores`, :func:`pooled_scores` and
    :func:`conditional_scores`). A discrete code of -1 marks a value never
    seen in training, which contributes nothing."""

    # Value of every discrete attribute as a child, shape (n, n_features)
    codes: np.ndarray
    # The Gaussian attributes, shape (n, n_gaussian)
    xg: np.ndarray
    # Log-likelihood of every attribute in a state never seen in training
    uninf: np.ndarray
    # Discretized codes of every attribute as a parent
    pcodes: np.ndarray
    # Whether the parent code of a value comes from a value never seen
    punseen: np.ndarray
