"""Compiled kernels of the attribute-conditional networks (TAN and KDB).

Factor ``m`` is ``P(x_j | y, x_pa)`` of child ``j = children[m]``. Its
augmented states are stored as those of an SPnDE; see
:class:`ConditionalTables`.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import numba as nb
import numpy as np

from ._runtime import (
    _BLOCK_ROWS,
    DISCRETE,
    GAUSSIAN,
    _find,
    _gaussian_ll,
    _Kernel,
)
from ._states import (
    _DENSE_CELLS_PER_ROW,
    _augmented_keys,
    _histogram,
    _state_index,
    _value_cells,
)


@nb.njit(cache=True)
def _sparse_pair_cmi(base, card, x, size, n_c, marg, off_i, ci, off_j):
    """``n * I(A_i; A_j | C)`` of a pair whose (c, a, b) table has ``size``
    cells, too many to count densely: the cells ``base[r] * card + x[r]`` of
    the samples are sorted, and the sum runs over the distinct ones in
    increasing order, the order in which the dense table visits its
    non-empty cells, so the result is the same."""
    n = base.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for r in range(n):
        keys[r] = base[r] * card + x[r]
    cells, index = _state_index(keys, size)
    counts = np.zeros(cells.shape[0], dtype=np.int64)
    for r in range(n):
        counts[index[r]] += 1
    total = 0.0
    for t in range(cells.shape[0]):
        ca, b = divmod(cells[t], card)
        c, a = divmod(ca, ci)
        n_cab = float(counts[t])
        ratio = n_cab * n_c[c] / (marg[off_i + c * ci + a] * marg[off_j + c * card + b])
        total += n_cab * np.log(ratio)
    return total


@_Kernel
def conditional_mutual_info(codes_t, y, cards, n_classes):
    """Empirical ``I(A_i; A_j | C)`` of every pair of attributes, in nats.

    ``codes_t`` holds the attribute codes transposed, shape (n_features, n).
    The (c, a) counts of every attribute are computed once; every pair counts
    its joint table and sums ``n_cab log(n_cab n_c / (n_ca n_cb))`` over its
    non-empty cells, which is exactly 0 for a constant attribute. A table
    with more than ``_DENSE_CELLS_PER_ROW`` cells per sample (attributes of
    thousands of values) is counted on its non-empty cells only.
    """
    d, n = codes_t.shape
    max_cells = _DENSE_CELLS_PER_ROW * n + 1024
    n_c = np.zeros(n_classes)
    for r in range(n):
        n_c[y[r]] += 1.0
    # (c, a) counts of attribute j at marg[moff[j] + c * cards[j] + a]
    moff = np.zeros(d + 1, dtype=np.int64)
    for j in range(d):
        moff[j + 1] = moff[j] + n_classes * cards[j]
    marg = np.zeros(moff[d])
    for j in nb.prange(d):
        for r in range(n):
            marg[moff[j] + y[r] * cards[j] + codes_t[j, r]] += 1.0
    cmi = np.zeros((d, d))
    for i in nb.prange(d):
        ci = cards[i]
        mi, base = moff[i], np.empty(n, dtype=np.int64)
        for r in range(n):
            base[r] = y[r] * ci + codes_t[i, r]
        dense = 0
        for j in range(i + 1, d):
            if n_classes * ci * cards[j] <= max_cells:
                dense = max(dense, n_classes * ci * cards[j])
        table = np.empty(dense, dtype=np.int64)
        for j in range(i + 1, d):
            cj, mj = cards[j], moff[j]
            size = n_classes * ci * cj
            if size > max_cells:
                total = _sparse_pair_cmi(
                    base, cj, codes_t[j], size, n_c, marg, mi, ci, mj
                )
                cmi[i, j] = cmi[j, i] = max(total / n, 0.0)
                continue
            # Integer counts, as count_pairs takes them
            _histogram(base, cj, codes_t[j], table, size)
            total = 0.0
            for c in range(n_classes):
                for a in range(ci):
                    row = (c * ci + a) * cj
                    for b in range(cj):
                        n_cab = float(table[row + b])
                        if n_cab > 0.0:
                            n_ca, n_cb = marg[mi + c * ci + a], marg[mj + c * cj + b]
                            ratio = n_cab * n_c[c] / (n_ca * n_cb)
                            total += n_cab * np.log(ratio)
            cmi[i, j] = cmi[j, i] = max(total / n, 0.0)
    return cmi


@_Kernel
def count_states(
    y, pcodes_t, parents, strides, n_combos, n_classes, codes, children, child_cards
):
    """Number of augmented states observed in training by each component
    and, for a component whose discrete child ``children[m]`` has
    ``child_cards[m]`` values and whose dense table (states x values) would
    have more than ``_DENSE_CELLS_PER_ROW`` cells per training row, the
    number of its non-empty (state, value) cells (0 for the others)."""
    n = y.shape[0]
    n_models = parents.shape[0]
    n_states = np.empty(n_models, dtype=np.int64)
    n_cells = np.zeros(n_models, dtype=np.int64)
    for m in nb.prange(n_models):
        keys = _augmented_keys(y, pcodes_t, parents[m], strides[m], n_combos[m])
        states, index = _state_index(keys, n_classes * n_combos[m])
        n_states[m] = states.shape[0]
        card = child_cards[m]
        if n_states[m] * card > _DENSE_CELLS_PER_ROW * n + 1024:
            cells = _value_cells(index, codes, children[m], card, n_states[m])
            n_cells[m] = cells[0].shape[0]
    return n_states, n_cells


@_Kernel
def count_conditionals(
    y,
    pcodes_t,
    codes,
    xg,
    parents,
    strides,
    children,
    n_combos,
    n_classes,
    state_off,
    disc_off,
    cell_off,
    kind,
    gidx,
    cards,
):
    """Sufficient statistics of every factor: the state keys, the training
    rows of every state, the value counts of a discrete child (on its
    non-empty cells, whose keys it also returns, when ``cell_off`` gives it
    some) and the mean and (biased) variance of a Gaussian one."""
    n = y.shape[0]
    n_factors = parents.shape[0]
    n_total = state_off[n_factors]
    keys_out = np.empty(n_total, dtype=np.int64)
    count = np.zeros(n_total)
    disc = np.zeros(disc_off[n_factors])
    cell_keys = np.empty(cell_off[n_factors], dtype=np.int64)
    mean = np.zeros(n_total)
    var = np.zeros(n_total)
    for m in nb.prange(n_factors):
        keys = _augmented_keys(y, pcodes_t, parents[m], strides[m], n_combos[m])
        states, index = _state_index(keys, n_classes * n_combos[m])
        o = state_off[m]
        keys_out[o : o + states.shape[0]] = states
        j = children[m]
        sparse = cell_off[m + 1] > cell_off[m]
        cell = index
        if sparse:
            cells, cell = _value_cells(index, codes, j, cards[j], states.shape[0])
            cell_keys[cell_off[m] : cell_off[m + 1]] = cells
        for r in range(n):
            s = o + index[r]
            count[s] += 1.0
            if kind[j] == DISCRETE:
                if sparse:
                    disc[disc_off[m] + cell[r]] += 1.0
                else:
                    disc[disc_off[m] + index[r] * cards[j] + codes[r, j]] += 1.0
            elif kind[j] == GAUSSIAN:
                mean[s] += xg[r, gidx[j]]
        if kind[j] != GAUSSIAN:
            continue
        for s in range(o, o + states.shape[0]):
            mean[s] /= count[s]
        for r in range(n):
            s = o + index[r]
            diff = xg[r, gidx[j]] - mean[s]
            var[s] += diff * diff
        for s in range(o, o + states.shape[0]):
            var[s] /= count[s]
    return keys_out, count, disc, cell_keys, mean, var


class ConditionalTables(NamedTuple):
    """The fitted conditionals of a TAN or KDB network (``tables`` of
    :func:`conditional_scores`).

    Factor ``m`` is ``P(x_j | y, x_pa)`` of child ``j = children[m]``; its
    augmented states ``s`` are ``state_off[m]:state_off[m + 1]``, and ``t``
    is the local state ``s - state_off[m]``. A discrete child whose dense
    table (states x values) would be too large, ``cell_off[m + 1] >
    cell_off[m]``, keeps only its non-empty cells.
    """

    # Parents of every factor, padded with stride 0, shape (n_factors, width)
    parents: np.ndarray
    # Stride of every parent in the parent key (the first most significant)
    strides: np.ndarray
    # Number of parents of every factor
    n_parents: np.ndarray
    # Child attribute of every factor
    children: np.ndarray
    # Number of parent instantiations of every factor
    n_combos: np.ndarray
    # Offset of the states of every factor, and their total at the end
    state_off: np.ndarray
    # Key ``y * n_combos + parent_key`` of every state, sorted per factor
    state_keys: np.ndarray
    # Offset of the log-probabilities of every factor in ``disc``
    disc_off: np.ndarray
    # Dense ``log P(x_j = v | s)`` at ``disc_off[m] + t * cards[j] + v``
    disc: np.ndarray
    # Factor ``m`` has the cells ``cell_off[m]:cell_off[m + 1]``
    cell_off: np.ndarray
    # Sorted keys ``t * cards[j] + v`` of the cells, values from disc_off[m]
    cell_keys: np.ndarray
    # Log-probability of a value never seen with state ``s`` of such a factor
    cell_unseen: np.ndarray
    # Mean of a Gaussian child in every state
    mean: np.ndarray
    # Its variance
    var: np.ndarray
    # ``log(2 pi var)``, from :func:`gaussian_log_norms` (0 for discrete)
    logn: np.ndarray
    # Type of every attribute: ``IGNORED``, ``DISCRETE`` or ``GAUSSIAN``
    kind: np.ndarray
    # Column of every Gaussian attribute in ``xg``
    gidx: np.ndarray
    # Number of values of every discrete attribute
    cards: np.ndarray


@_Kernel
def conditional_scores(inputs, tables, n_classes, per_factor):
    """Sum over the factors of ``log P(x_j | y, x_pa)``, shape
    (n, n_classes, 1), from the :class:`EvaluationInputs` of the samples and
    the :class:`ConditionalTables` of the network. With ``per_factor`` the factors are kept apart instead,
    factor ``m`` at ``[:, :, 1 + m]``, and ``[:, :, 0]`` is left at 0 for the
    class log-prior, which the caller adds.

    A state never seen in training, or a parent value never seen, gives the
    uninformed estimate ``uninf[r, j]``.
    """
    # The fields are read once, by name: the loops below index plain arrays
    codes = inputs.codes
    xg = inputs.xg
    uninf = inputs.uninf
    pcodes = inputs.pcodes
    punseen = inputs.punseen
    parents = tables.parents
    strides = tables.strides
    n_parents = tables.n_parents
    children = tables.children
    n_combos = tables.n_combos
    state_off = tables.state_off
    state_keys = tables.state_keys
    disc_off = tables.disc_off
    disc = tables.disc
    cell_off = tables.cell_off
    cell_keys = tables.cell_keys
    cell_unseen = tables.cell_unseen
    mean = tables.mean
    var = tables.var
    logn = tables.logn
    kind = tables.kind
    gidx = tables.gidx
    cards = tables.cards

    n = codes.shape[0]
    n_factors = parents.shape[0]
    out = np.zeros((n, n_classes, 1 + n_factors if per_factor else 1))
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        for m in range(n_factors):
            j = children[m]
            o = state_off[m]
            lo_cell, hi_cell = cell_off[m], cell_off[m + 1]
            col = 1 + m if per_factor else 0
            for r in range(lo, hi):
                key = 0
                unseen = False
                for q in range(n_parents[m]):
                    p = parents[m, q]
                    key += pcodes[r, p] * strides[m, q]
                    unseen |= punseen[r, p]
                for c in range(n_classes):
                    s = -1
                    if not unseen:
                        s = _find(
                            state_keys, o, state_off[m + 1], c * n_combos[m] + key
                        )
                    if s < 0:
                        out[r, c, col] += uninf[r, j]
                    elif kind[j] == DISCRETE:
                        code = codes[r, j]
                        cell = (s - o) * cards[j] + code
                        if code >= 0 and hi_cell == lo_cell:
                            out[r, c, col] += disc[disc_off[m] + cell]
                        elif code >= 0:
                            t = _find(cell_keys, lo_cell, hi_cell, cell)
                            if t < 0:
                                out[r, c, col] += cell_unseen[s]
                            else:
                                out[r, c, col] += disc[disc_off[m] + t - lo_cell]
                    elif kind[j] == GAUSSIAN:
                        out[r, c, col] += _gaussian_ll(
                            xg[r, gidx[j]], mean[s], var[s], logn[s]
                        )
    return out
