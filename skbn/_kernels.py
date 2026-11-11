"""Compiled kernels that count and evaluate the SPnDEs of an ensemble.

Every SPnDE is a Naive Bayes model on the augmented class ``Y* = (y, x_p)``.
Its parameters are stored per augmented state observed in training, in flat
tables shared by all the components of the ensemble, described field by
field in :class:`SPnDETables`. The conditionals of TAN and KDB are stored the
same way (:class:`ConditionalTables`), and the samples to evaluate come in an
:class:`EvaluationInputs`.

The kernels of the SPnDEs take the named tuples themselves and read their
fields by name. :func:`conditional_scores` takes them splatted
(``*tables``), so the order of their fields is the order of its
parameters, which a test checks.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import threading
import types
from contextlib import contextmanager
from typing import NamedTuple

import numba as nb
import numpy as np
from joblib import effective_n_jobs

IGNORED, DISCRETE, GAUSSIAN = 0, 1, 2

# Value that np.nan_to_num(nan=-1e10, neginf=-1e10) gives a degenerate
# log-likelihood (zero probability or variance)
LOG_ZERO = -1e10
_FLOAT_MAX = np.finfo(np.float64).max

# Augmented-state grids up to this many cells per training row are indexed
# with a dense array; larger ones are sorted.
_DENSE_CELLS_PER_ROW = 4

# Augmented states with up to this many training rows keep the rows instead of
# a dense row of estimates
_MAX_LIGHT = 16

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
def index_states(y, pcodes_t, parents, strides, n_combos, n_classes, max_light):
    """Sizes of the tables of every component, shape (n_models, 3): its
    number of augmented states seen in training, of dense states (more than
    ``max_light`` training rows) and of training rows in its light states."""
    n = y.shape[0]
    n_models = parents.shape[0]
    sizes = np.zeros((n_models, 3), dtype=np.int64)
    for m in nb.prange(n_models):
        keys = _augmented_keys(y, pcodes_t, parents[m], strides[m], n_combos[m])
        states, index = _state_index(keys, n_classes * n_combos[m])
        rows_of = np.zeros(states.shape[0], dtype=np.int64)
        for r in range(n):
            rows_of[index[r]] += 1
        sizes[m, 0] = states.shape[0]
        for s in range(states.shape[0]):
            if rows_of[s] > max_light:
                sizes[m, 1] += 1
            else:
                sizes[m, 2] += rows_of[s]
    return sizes


@_Kernel
def count_tables(
    y,
    pcodes_t,
    codes,
    xg,
    parents,
    strides,
    n_combos,
    n_classes,
    max_light,
    state_off,
    dense_off,
    light_off,
    attrs,
    kind,
    doff,
    gidx,
    n_columns,
    by_shared,
    pcards,
    max_cells,
):
    """Sufficient statistics of every component.

    Returns the state keys, the number of training rows of every state, the
    dense row of every state (-1 for a light one) and the offset of its
    training rows in ``rows`` (-1 for a dense one), the value counts of the
    dense states (``n_columns`` columns, see ``doff``), the training rows of
    the light states, and the mean and the (biased) variance of every
    Gaussian attribute per state. ``codes`` holds the value codes of the
    training rows. The columns of the parents of a component are left at
    zero, and so are those of the discrete children that :func:`count_pairs`
    (one parent) or :func:`count_triples` (two) count: when ``by_shared[m]``,
    the children ``j`` whose table over the class, the parents and ``j`` has
    at most ``max_cells`` cells (``n_classes * pcards[j]`` times the
    ``pcards`` of the parents).
    """
    n = y.shape[0]
    n_models = parents.shape[0]
    n_dep = parents.shape[1]
    n_total = state_off[n_models]
    n_gauss = xg.shape[1]
    keys_out = np.empty(n_total, dtype=np.int64)
    count = np.zeros(n_total)
    drow = np.empty(n_total, dtype=np.int64)
    loff = np.empty(n_total, dtype=np.int64)
    dense = np.zeros((dense_off[n_models], n_columns))
    rows = np.empty(light_off[n_models], dtype=np.int32)
    mean = np.zeros((n_total, n_gauss))
    var = np.zeros((n_total, n_gauss))
    for m in nb.prange(n_models):
        keys = _augmented_keys(y, pcodes_t, parents[m], strides[m], n_combos[m])
        states, index = _state_index(keys, n_classes * n_combos[m])
        o = state_off[m]
        n_states = states.shape[0]
        keys_out[o : o + n_states] = states
        for r in range(n):
            count[o + index[r]] += 1.0
        next_dense, next_light = dense_off[m], light_off[m]
        for s in range(o, o + n_states):
            if count[s] > max_light:
                drow[s], loff[s] = next_dense, -1
                next_dense += 1
            else:
                drow[s], loff[s] = -1, next_light
                next_light += int(count[s])
        # Training rows of the light states, in row order; dense row of every
        # training row (-1 when its state is light)
        filled = np.zeros(n_states, dtype=np.int64)
        row_dense = np.empty(n, dtype=np.int64)
        for r in range(n):
            s = o + index[r]
            row_dense[r] = drow[s]
            if drow[s] < 0:
                rows[loff[s] + filled[index[r]]] = r
                filled[index[r]] += 1
        # Discrete children of the dense states
        child = np.empty(attrs.shape[0], dtype=np.int64)
        n_child = 0
        q = 0
        for a in range(attrs.shape[0]):
            j = attrs[a]
            while q < n_dep and parents[m, q] < j:
                q += 1
            if (q < n_dep and parents[m, q] == j) or kind[j] != DISCRETE:
                continue
            if by_shared[m]:
                cells = n_classes * pcards[j]
                for p in range(n_dep):
                    cells *= pcards[parents[m, p]]
                if cells <= max_cells:
                    continue  # Counted by count_pairs or count_triples
            child[n_child] = j
            n_child += 1
        offset = np.empty(n_child, dtype=np.int64)
        for t in range(n_child):
            offset[t] = doff[child[t]]
        for r in range(n):
            if row_dense[r] < 0:
                continue
            row = dense[row_dense[r]]
            sample = codes[r]
            for t in range(n_child):
                row[offset[t] + sample[child[t]]] += 1.0
        if n_gauss == 0:
            continue
        # Gaussian children of every state
        q = 0
        for a in range(attrs.shape[0]):
            j = attrs[a]
            while q < n_dep and parents[m, q] < j:
                q += 1
            if (q < n_dep and parents[m, q] == j) or kind[j] != GAUSSIAN:
                continue
            g = gidx[j]
            for r in range(n):
                mean[o + index[r], g] += xg[r, g]
            for s in range(o, o + n_states):
                mean[s, g] /= count[s]
            for r in range(n):
                diff = xg[r, g] - mean[o + index[r], g]
                var[o + index[r], g] += diff * diff
            for s in range(o, o + n_states):
                var[s, g] /= count[s]
    return keys_out, count, drow, loff, dense, rows, mean, var


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


@_Kernel
def count_pairs(
    codes_t, y, n_classes, cards, single, child_col, smap, moff, max_cells, dense
):
    """Value counts of the discrete children of the components with one
    parent, in the rows of their dense states, from one count of every pair
    of attributes: :func:`count_triples` for one parent.

    ``single[a]`` is the component of parent ``a``, whose state
    ``y * cards[a] + x_a`` has its dense row in ``smap[moff[single[a]] +
    state]``. The counts ``N(y, x_a, x_b)`` of the pair ``a < b`` give the
    child ``b`` of ``(a)`` and the child ``a`` of ``(b)``. A pair with more
    than ``max_cells`` cells is left to :func:`count_tables`.
    """
    n_attrs, n = codes_t.shape
    for a in nb.prange(n_attrs - 1):
        ca = cards[a]
        xa = codes_t[a]
        base = np.empty(n, dtype=np.int64)
        for r in range(n):
            base[r] = y[r] * ca + xa[r]
        cells = np.empty(max_cells, dtype=np.int64)
        m_a = moff[single[a]]
        col_a = child_col[a]
        for b in range(a + 1, n_attrs):
            cb = cards[b]
            size = n_classes * ca * cb
            if size > max_cells:
                continue  # Counted by count_tables
            xb = codes_t[b]
            m_b, col_b = moff[single[b]], child_col[b]
            counts = cells
            _histogram(base, cb, xb, counts, size)
            # Cell (y * ca + va) * cb + vb
            for k in range(n_classes):
                for va in range(ca):
                    block = (k * ca + va) * cb
                    if col_b >= 0:
                        row = smap[m_a + k * ca + va]
                        if row >= 0:
                            for vb in range(cb):
                                dense[row, col_b + vb] = counts[block + vb]
                    if col_a >= 0:
                        for vb in range(cb):
                            row = smap[m_b + k * cb + vb]
                            if row >= 0:
                                dense[row, col_a + va] = counts[block + vb]


@_Kernel
def count_triples(
    codes_t,
    y,
    n_classes,
    cards,
    first,
    second,
    pair_group,
    child_col,
    smap,
    moff,
    max_cells,
    dense,
):
    """Value counts of the discrete children of the components with two
    parents, in the rows of their dense states, from one count of every
    triple of attributes.

    The attributes are those of ``codes_t``, shape (n_attrs, n), with
    ``cards[i]`` values; ``pair_group[a, b]`` is the component of parents
    ``a < b``, whose state ``(y * cards[a] + x_a) * cards[b] + x_b`` has its
    dense row in ``smap[moff[pair_group[a, b]] + state]`` (-1 for a light or
    unseen state), and ``child_col[c]`` is the column of the values of ``c``
    as a child (-1 when it is not a discrete child). The counts
    ``N(y, x_a, x_b, x_c)`` of the triple ``a < b < c`` give the children
    ``c``, ``b`` and ``a`` of the components ``(a, b)``, ``(a, c)`` and
    ``(b, c)``: one pass over the samples instead of three. A triple with
    more than ``max_cells`` cells is left to :func:`count_tables`, which
    counts the rows of the dense states only.
    The pairs ``(first[t], second[t])`` are processed in parallel; each
    child column of a component belongs to a single triple.
    """
    n_attrs, n = codes_t.shape
    for t in nb.prange(first.shape[0]):
        a, b = first[t], second[t]
        ca, cb = cards[a], cards[b]
        xa, xb = codes_t[a], codes_t[b]
        base = np.empty(n, dtype=np.int64)
        for r in range(n):
            base[r] = (y[r] * ca + xa[r]) * cb + xb[r]
        cells = np.empty(max_cells, dtype=np.int64)
        m_ab = moff[pair_group[a, b]]
        for c in range(b + 1, n_attrs):
            cc = cards[c]
            xc = codes_t[c]
            m_ac, m_bc = moff[pair_group[a, c]], moff[pair_group[b, c]]
            col_a, col_b, col_c = child_col[a], child_col[b], child_col[c]
            size = n_classes * ca * cb * cc
            if size > max_cells:
                continue  # Counted by count_tables
            counts = cells
            _histogram(base, cc, xc, counts, size)
            # Cell ((y * ca + va) * cb + vb) * cc + vc
            for k in range(n_classes):
                for va in range(ca):
                    for vb in range(cb):
                        block = ((k * ca + va) * cb + vb) * cc
                        if col_c >= 0:
                            row = smap[m_ab + (k * ca + va) * cb + vb]
                            if row >= 0:
                                for vc in range(cc):
                                    dense[row, col_c + vc] = counts[block + vc]
                        for vc in range(cc):
                            if col_b >= 0:
                                row = smap[m_ac + (k * ca + va) * cc + vc]
                                if row >= 0:
                                    dense[row, col_b + vb] = counts[block + vc]
                            if col_a >= 0:
                                row = smap[m_bc + (k * cb + vb) * cc + vc]
                                if row >= 0:
                                    dense[row, col_a + va] = counts[block + vc]


@_Kernel
def dense_estimates(dense, log_hits, log_total, group_of):
    """Replace, in place, the value counts ``k = dense[s, c]`` of the dense
    states by ``log_hits[k] - log_total[s, group_of[c]]``, clamped as
    ``np.nan_to_num``. With the logarithms taken by NumPy over every count
    and every state total, the estimates are those of NumPy over the whole
    table, without a logarithm per entry."""
    for s in nb.prange(dense.shape[0]):
        row = dense[s]
        total = log_total[s]
        for c in range(row.shape[0]):
            row[c] = _clamp(log_hits[int(row[c])] - total[group_of[c]])


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


@nb.njit(cache=True)
def _log_likelihood(s, j, r, codes, xg, disc, mean, var, logn, kind, voff, gidx):
    """``log P(x_rj | state s)``, clamped as ``np.nan_to_num`` (the discrete
    tables are clamped when they are built).

    Used on the back-off paths that few samples take. It is called, not
    inlined: inlined at the level of Numba's IR it made the loops of the
    evaluation kernels 20 to 30 times slower, so the hot loops read the
    tables directly."""
    if kind[j] == DISCRETE:
        code = codes[r, j]
        if code < 0:
            return 0.0
        return disc[s, voff[j] + code]
    if kind[j] == GAUSSIAN:
        g = gidx[j]
        return _gaussian_ll(xg[r, g], mean[s, g], var[s, g], logn[s, g])
    return 0.0


@nb.njit(cache=True)
def _light_buffers(codes, lo, hi, attrs, dcol, tcodes):
    """Codes of the samples ``lo:hi`` in the column order of ``tcodes``, and
    a buffer for the hits of a light state in every column."""
    sample_codes = np.empty((hi - lo, tcodes.shape[1]), dtype=np.int64)
    for a in range(attrs.shape[0]):
        j = attrs[a]
        if dcol[j] >= 0:
            for r in range(lo, hi):
                sample_codes[r - lo, dcol[j]] = codes[r, j]
    return sample_codes, np.empty(tcodes.shape[1], dtype=np.int64)


@nb.njit(cache=True, inline="always")
def _count_hits(s, count, loff, rows, tcodes, sample, hits):
    """Number of training rows of the light state ``s`` that share the value
    of ``sample`` in every column of ``tcodes``. The rows are read whole, one
    after the other; read column by column, each row would be loaded once
    per child."""
    hits[:] = 0
    for t in range(loff[s], loff[s] + int(count[s])):
        trow = tcodes[rows[t]]
        for col in range(hits.shape[0]):
            hits[col] += trow[col] == sample[col]


@nb.njit(cache=True, inline="always")
def _parent_state(
    r, m, g, pcodes, punseen, parents, strides, pk_off, pk_keys, pk_count, skip
):
    """Parent key of component ``m`` (of group ``g``) for row ``r`` and the
    training frequency of that parent instantiation (0 when the value of a
    parent ``p`` with ``not skip[p]`` was never seen)."""
    key = 0
    unseen = False
    for q in range(parents.shape[1]):
        p = parents[m, q]
        key += pcodes[r, p] * strides[m, q]
        unseen |= punseen[r, p] and not skip[p]
    pos = _find(pk_keys, pk_off[g], pk_off[g + 1], key)
    freq = 0.0 if (pos < 0 or unseen) else pk_count[pos]
    return key, freq


@nb.njit(cache=True)
def _class_children(
    r,
    m,
    states,
    rows_of,
    totals,
    class_hits,
    codes,
    xg,
    parents,
    count,
    dense,
    lut,
    mean,
    var,
    logn,
    kind,
    gidx,
    attrs,
    dcol,
    doff,
    grp,
):
    """Add the children of component ``m`` for sample ``r`` to ``totals``,
    for every class whose state ``states[c]`` was seen (dense row
    ``rows_of[c]``, hits ``class_hits[c]`` of a light one). Each child is read
    once for all the classes, and each class sums its children in order."""
    n_dep = parents.shape[1]
    n_classes = states.shape[0]
    q = 0
    for a in range(attrs.shape[0]):
        j = attrs[a]
        while q < n_dep and parents[m, q] < j:
            q += 1
        if q < n_dep and parents[m, q] == j:
            continue
        if kind[j] == DISCRETE:
            code = codes[r, j]
            if code < 0:
                continue
            column, gj, cj = doff[j] + code, grp[j], dcol[j]
            for c in range(n_classes):
                st = states[c]
                if st < 0:
                    continue
                if rows_of[c] >= 0:
                    totals[c] += dense[rows_of[c], column]
                else:
                    totals[c] += lut[int(count[st]), gj, class_hits[c, cj]]
        else:
            g = gidx[j]
            x = xg[r, g]
            for c in range(n_classes):
                st = states[c]
                if st >= 0:
                    totals[c] += _gaussian_ll(x, mean[st, g], var[st, g], logn[st, g])


@nb.njit(cache=True)
def _two_class_children(
    r,
    m,
    states,
    rows_of,
    totals,
    class_hits,
    codes,
    xg,
    parents,
    count,
    dense,
    lut,
    mean,
    var,
    logn,
    kind,
    gidx,
    attrs,
    dcol,
    doff,
    grp,
):
    """:func:`_class_children` for two classes, the sums kept in registers."""
    n_dep = parents.shape[1]
    s0, s1 = states[0], states[1]
    d0, d1 = rows_of[0], rows_of[1]
    t0, t1 = totals[0], totals[1]
    q = 0
    for a in range(attrs.shape[0]):
        j = attrs[a]
        while q < n_dep and parents[m, q] < j:
            q += 1
        if q < n_dep and parents[m, q] == j:
            continue
        if kind[j] == DISCRETE:
            code = codes[r, j]
            if code < 0:
                continue
            column, gj, cj = doff[j] + code, grp[j], dcol[j]
            if s0 >= 0:
                if d0 >= 0:
                    t0 += dense[d0, column]
                else:
                    t0 += lut[int(count[s0]), gj, class_hits[0, cj]]
            if s1 >= 0:
                if d1 >= 0:
                    t1 += dense[d1, column]
                else:
                    t1 += lut[int(count[s1]), gj, class_hits[1, cj]]
        else:
            g = gidx[j]
            x = xg[r, g]
            if s0 >= 0:
                t0 += _gaussian_ll(x, mean[s0, g], var[s0, g], logn[s0, g])
            if s1 >= 0:
                t1 += _gaussian_ll(x, mean[s1, g], var[s1, g], logn[s1, g])
    totals[0], totals[1] = t0, t1


class EvaluationInputs(NamedTuple):
    """Per-sample inputs of the evaluation kernels: ``inputs`` of
    :func:`component_scores` and :func:`pooled_scores`, and the first
    parameters of :func:`conditional_scores`. A discrete code of -1 marks a
    value never seen in training, which contributes nothing."""

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


class SPnDETables(NamedTuple):
    """The fitted tables of an ensemble of SPnDEs, ``tables`` of
    :func:`component_scores` and :func:`pooled_scores`.

    The components that differ only in which constant attributes
    (``const``) are among their parents have the same parameters, stored
    once for their group ``g = group[m]``; its augmented states ``s`` are
    ``state_off[g]:state_off[g + 1]``. ``log P(x_j = v | s)`` of a discrete
    child (categorical, or Bernoulli with values 0 and 1) has ``-inf``
    replaced by ``LOG_ZERO``. A dense state (more than ``_MAX_LIGHT``
    training rows) keeps it in a row of ``dense``; a light state keeps only
    its training rows, and the number ``k`` of them with ``x_j = v`` gives
    the estimate from ``lut``. Most states of high-order or high-cardinality
    ensembles have a few rows, and a dense row per state would cost far more
    memory.
    """

    # Super-parents of every component, shape (n_models, n_dependence)
    parents: np.ndarray
    # Stride of every parent in the parent key (the first most significant)
    strides: np.ndarray
    # Number of parent instantiations of every component
    n_combos: np.ndarray
    # Group of every component
    group: np.ndarray
    # First component of every group, which computes its scores
    rep: np.ndarray
    # Attributes with one value as parents and no information as children
    const: np.ndarray
    # Offset of the states of every group, and their total at the end
    state_off: np.ndarray
    # Key ``y * n_combos + parent_key`` of every state, sorted per group
    state_keys: np.ndarray
    # Root ``log P(y, x_p)`` of every state
    root: np.ndarray
    # Number of training rows of every state
    count: np.ndarray
    # Row of every dense state in ``dense``, -1 for a light one
    drow: np.ndarray
    # Light state ``s`` has the rows ``rows[loff[s]:loff[s] + count[s]]``
    loff: np.ndarray
    # ``log P(x_j = v | s)`` of the dense states, at column ``doff[j] + v``
    dense: np.ndarray
    # Row in ``tcodes`` of every training row of the light states
    rows: np.ndarray
    # Value codes of those training rows, attribute ``j`` in column dcol[j]
    tcodes: np.ndarray
    # Light estimate ``lut[count[s], grp[j], k]``
    lut: np.ndarray
    # ``mean[s, g]`` of the Gaussian of attribute ``j`` (``g = gidx[j]``)
    mean: np.ndarray
    # ``var[s, g]``, its variance
    var: np.ndarray
    # ``log(2 pi var)``, from :func:`gaussian_log_norms`
    logn: np.ndarray
    # Group ``g`` has the parent keys ``pk_off[g]:pk_off[g + 1]``
    pk_off: np.ndarray
    # Distinct parent keys seen in training, sorted per group
    pk_keys: np.ndarray
    # Training frequency of every parent key
    pk_count: np.ndarray
    # Root of a state never seen in training, per group
    log_prior_unseen: np.ndarray
    # Class log-prior of the class-level back-off Naive Bayes
    b_root: np.ndarray
    # Its discrete log-conditionals, of value ``v`` of ``j`` at ``voff[j] + v``
    b_disc: np.ndarray
    # Its Gaussian means, shape (n_classes, n_gaussian)
    b_mean: np.ndarray
    # Its Gaussian variances
    b_var: np.ndarray
    # Their ``log(2 pi var)``
    b_logn: np.ndarray
    # Type of every attribute: ``IGNORED``, ``DISCRETE`` or ``GAUSSIAN``
    kind: np.ndarray
    # Offset of every discrete attribute in the columns of ``b_disc``
    voff: np.ndarray
    # Column of every Gaussian attribute in ``xg``, ``mean`` and ``var``
    gidx: np.ndarray
    # Possible children: Gaussian or discrete with more than one value
    attrs: np.ndarray
    # Column of every discrete child in ``tcodes``, -1 for the others
    dcol: np.ndarray
    # Offset of every discrete child in the columns of ``dense``
    doff: np.ndarray
    # Index of the number of values of every attribute in ``lut``
    grp: np.ndarray
    # Attributes whose estimate in an unseen state can differ from 0
    nonzero: np.ndarray


@_Kernel
def component_scores(inputs, tables, min_count, backoff, per_factor):
    """Log-joint ``log P_m(y, x)`` of every sample, class and component, from
    the :class:`EvaluationInputs` of the samples and the
    :class:`SPnDETables` of the ensemble.

    Returns ``out`` of shape (n, n_classes, n_models), or with
    ``per_factor`` of shape (n, n_classes, n_models, 1 + n_children) holding
    the root and child factors; ``active`` of shape (n, n_models); and the
    Naive Bayes joint of the samples with no active component (zero for the
    others).
    """
    # The fields are read once, by name: the loops below index plain arrays
    codes = inputs.codes
    xg = inputs.xg
    uninf = inputs.uninf
    pcodes = inputs.pcodes
    punseen = inputs.punseen
    parents = tables.parents
    strides = tables.strides
    n_combos = tables.n_combos
    group = tables.group
    state_off = tables.state_off
    state_keys = tables.state_keys
    root = tables.root
    count = tables.count
    drow = tables.drow
    loff = tables.loff
    dense = tables.dense
    rows = tables.rows
    tcodes = tables.tcodes
    lut = tables.lut
    mean = tables.mean
    var = tables.var
    logn = tables.logn
    pk_off = tables.pk_off
    pk_keys = tables.pk_keys
    pk_count = tables.pk_count
    log_prior_unseen = tables.log_prior_unseen
    b_root = tables.b_root
    b_disc = tables.b_disc
    b_mean = tables.b_mean
    b_var = tables.b_var
    b_logn = tables.b_logn
    kind = tables.kind
    voff = tables.voff
    gidx = tables.gidx
    attrs = tables.attrs
    dcol = tables.dcol
    doff = tables.doff
    grp = tables.grp
    nonzero = tables.nonzero

    n, n_features = codes.shape
    n_models, n_dep = parents.shape
    n_classes = b_root.shape[0]
    n_factors = 1 + n_features - n_dep if per_factor else 1
    out = np.zeros((n, n_classes, n_models, n_factors))
    active = np.empty((n, n_models), dtype=np.bool_)
    nb_jll = np.zeros((n, n_classes))
    no_skip = np.zeros(n_features, dtype=np.bool_)
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        sample_codes, hits = _light_buffers(codes, lo, hi, attrs, dcol, tcodes)
        class_hits = np.empty((n_classes, hits.shape[0]), dtype=np.int64)
        states = np.empty(n_classes, dtype=np.int64)
        rows_of = np.empty(n_classes, dtype=np.int64)
        totals = np.empty(n_classes)
        for m in range(n_models):
            grp_m = group[m]
            for r in range(lo, hi):
                key, freq = _parent_state(
                    r,
                    m,
                    grp_m,
                    pcodes,
                    punseen,
                    parents,
                    strides,
                    pk_off,
                    pk_keys,
                    pk_count,
                    no_skip,
                )
                act = freq >= min_count or n_dep == 0
                active[r, m] = act
                if not per_factor:
                    # As in pooled_scores: the states of every class, then
                    # each child read once for all of them
                    for c in range(n_classes):
                        st = _find(
                            state_keys,
                            state_off[grp_m],
                            state_off[grp_m + 1],
                            c * n_combos[m] + key,
                        )
                        states[c] = st
                        if st >= 0:
                            rows_of[c] = drow[st]
                            totals[c] = root[st]
                            if drow[st] < 0:
                                _count_hits(
                                    st,
                                    count,
                                    loff,
                                    rows,
                                    tcodes,
                                    sample_codes[r - lo],
                                    class_hits[c],
                                )
                    if n_classes == 2:
                        _two_class_children(
                            r,
                            m,
                            states,
                            rows_of,
                            totals,
                            class_hits,
                            codes,
                            xg,
                            parents,
                            count,
                            dense,
                            lut,
                            mean,
                            var,
                            logn,
                            kind,
                            gidx,
                            attrs,
                            dcol,
                            doff,
                            grp,
                        )
                    else:
                        _class_children(
                            r,
                            m,
                            states,
                            rows_of,
                            totals,
                            class_hits,
                            codes,
                            xg,
                            parents,
                            count,
                            dense,
                            lut,
                            mean,
                            var,
                            logn,
                            kind,
                            gidx,
                            attrs,
                            dcol,
                            doff,
                            grp,
                        )
                    for c in range(n_classes):
                        if states[c] >= 0:
                            out[r, c, m, 0] = totals[c]
                        elif act:
                            total = log_prior_unseen[grp_m]
                            q = 0
                            for a in range(nonzero.shape[0]):
                                j = nonzero[a]
                                while q < n_dep and parents[m, q] < j:
                                    q += 1
                                if q < n_dep and parents[m, q] == j:
                                    continue
                                if backoff:
                                    total += _log_likelihood(
                                        c,
                                        j,
                                        r,
                                        codes,
                                        xg,
                                        b_disc,
                                        b_mean,
                                        b_var,
                                        b_logn,
                                        kind,
                                        voff,
                                        gidx,
                                    )
                                else:
                                    total += uninf[r, j]
                            out[r, c, m, 0] = total
                    continue
                for c in range(n_classes):
                    s = _find(
                        state_keys,
                        state_off[grp_m],
                        state_off[grp_m + 1],
                        c * n_combos[m] + key,
                    )
                    if s >= 0:
                        # Children with a single value, or a value unseen in
                        # training, contribute 0 and are skipped
                        if drow[s] < 0:
                            _count_hits(
                                s, count, loff, rows, tcodes, sample_codes[r - lo], hits
                            )
                        total = root[s]
                        out[r, c, m, 0] = total
                        q = 0
                        for a in range(attrs.shape[0]):
                            j = attrs[a]
                            while q < n_dep and parents[m, q] < j:
                                q += 1
                            if q < n_dep and parents[m, q] == j:
                                continue
                            if kind[j] == DISCRETE:
                                code = codes[r, j]
                                if code < 0:
                                    continue
                                if drow[s] >= 0:
                                    ll = dense[drow[s], doff[j] + code]
                                else:
                                    ll = lut[int(count[s]), grp[j], hits[dcol[j]]]
                            else:
                                g = gidx[j]
                                ll = _gaussian_ll(
                                    xg[r, g], mean[s, g], var[s, g], logn[s, g]
                                )
                            if per_factor:
                                out[r, c, m, 1 + j - q] = ll
                            else:
                                total += ll
                        if not per_factor:
                            out[r, c, m, 0] = total
                    elif act:
                        total = log_prior_unseen[grp_m]
                        out[r, c, m, 0] = total
                        q = 0
                        for a in range(nonzero.shape[0]):
                            j = nonzero[a]
                            while q < n_dep and parents[m, q] < j:
                                q += 1
                            if q < n_dep and parents[m, q] == j:
                                continue
                            if backoff:
                                ll = _log_likelihood(
                                    c,
                                    j,
                                    r,
                                    codes,
                                    xg,
                                    b_disc,
                                    b_mean,
                                    b_var,
                                    b_logn,
                                    kind,
                                    voff,
                                    gidx,
                                )
                            else:
                                ll = uninf[r, j]
                            if per_factor:
                                out[r, c, m, 1 + j - q] = ll
                            else:
                                total += ll
                        if not per_factor:
                            out[r, c, m, 0] = total
        for r in range(lo, hi):
            if active[r].any():
                continue
            for c in range(n_classes):
                total = b_root[c]
                for a in range(nonzero.shape[0]):
                    total += _log_likelihood(
                        c,
                        nonzero[a],
                        r,
                        codes,
                        xg,
                        b_disc,
                        b_mean,
                        b_var,
                        b_logn,
                        kind,
                        voff,
                        gidx,
                    )
                nb_jll[r, c] = total
    return out, active, nb_jll


@_Kernel
def pooled_scores(inputs, tables, min_count, backoff, geometric):
    """Pooled score of every sample and class, shape (n, n_classes), from the
    :class:`EvaluationInputs` of the samples and the :class:`SPnDETables` of
    the ensemble.

    The arithmetic pool is ``log mean_m P_m(y, x)`` and the geometric one
    ``mean_m log P_m(y, x)``, over the active components; samples with no
    active component get the Naive Bayes joint. Equals pooling the output of
    :func:`component_scores` without materialising it.

    The components of a group (``group[m]``) differ only in which attributes
    with a single value (``const``) are among their parents, and give the
    same score to every sample. The score is computed once per group, by its
    first component ``rep[g]``, and pooled for every component, in their
    order; a component is left out of the pool of a sample whose value of
    one of its constant parents was never seen.
    """
    # The fields are read once, by name: the loops below index plain arrays
    codes = inputs.codes
    xg = inputs.xg
    uninf = inputs.uninf
    pcodes = inputs.pcodes
    punseen = inputs.punseen
    parents = tables.parents
    strides = tables.strides
    n_combos = tables.n_combos
    group = tables.group
    rep = tables.rep
    const = tables.const
    state_off = tables.state_off
    state_keys = tables.state_keys
    root = tables.root
    count = tables.count
    drow = tables.drow
    loff = tables.loff
    dense = tables.dense
    rows = tables.rows
    tcodes = tables.tcodes
    lut = tables.lut
    mean = tables.mean
    var = tables.var
    logn = tables.logn
    pk_off = tables.pk_off
    pk_keys = tables.pk_keys
    pk_count = tables.pk_count
    log_prior_unseen = tables.log_prior_unseen
    b_root = tables.b_root
    b_disc = tables.b_disc
    b_mean = tables.b_mean
    b_var = tables.b_var
    b_logn = tables.b_logn
    kind = tables.kind
    voff = tables.voff
    gidx = tables.gidx
    attrs = tables.attrs
    dcol = tables.dcol
    doff = tables.doff
    grp = tables.grp
    nonzero = tables.nonzero

    n, n_features = codes.shape
    n_models, n_dep = parents.shape
    n_groups = rep.shape[0]
    n_classes = b_root.shape[0]
    scores = np.empty((n, n_classes))
    # With groups, the arithmetic pool keeps the term exp(score - peak) of
    # every group and sample (see below)
    reuse = n_groups < n_models and not geometric
    # Samples per block, so that the scores (and terms) of the groups stay
    # within 32 MB
    size = n_groups * n_classes * (3 if reuse else 1)
    block = max(1, min(_BLOCK_ROWS, (1 << 22) // size))
    n_blocks = (n + block - 1) // block
    for b in nb.prange(n_blocks):
        lo, hi = b * block, min(n, (b + 1) * block)
        group_score = np.empty((n_groups, hi - lo, n_classes))
        group_active = np.empty((n_groups, hi - lo), dtype=np.bool_)
        sample_codes, hits = _light_buffers(codes, lo, hi, attrs, dcol, tcodes)
        class_hits = np.empty((n_classes, hits.shape[0]), dtype=np.int64)
        states = np.empty(n_classes, dtype=np.int64)
        rows_of = np.empty(n_classes, dtype=np.int64)
        totals = np.empty(n_classes)
        for grp_m in range(n_groups):
            m = rep[grp_m]
            for r in range(lo, hi):
                i = r - lo
                key, freq = _parent_state(
                    r,
                    m,
                    grp_m,
                    pcodes,
                    punseen,
                    parents,
                    strides,
                    pk_off,
                    pk_keys,
                    pk_count,
                    const,
                )
                act = freq >= min_count or n_dep == 0
                group_active[grp_m, i] = act
                if not act:
                    continue
                # The states of every class, then their children read once
                # for all the classes: the sums of the classes, each in the
                # order of the children, do not wait on each other
                for c in range(n_classes):
                    st = _find(
                        state_keys,
                        state_off[grp_m],
                        state_off[grp_m + 1],
                        c * n_combos[m] + key,
                    )
                    states[c] = st
                    if st >= 0:
                        rows_of[c] = drow[st]
                        totals[c] = root[st]
                        if drow[st] < 0:
                            _count_hits(
                                st,
                                count,
                                loff,
                                rows,
                                tcodes,
                                sample_codes[i],
                                class_hits[c],
                            )
                if n_classes == 2:
                    _two_class_children(
                        r,
                        m,
                        states,
                        rows_of,
                        totals,
                        class_hits,
                        codes,
                        xg,
                        parents,
                        count,
                        dense,
                        lut,
                        mean,
                        var,
                        logn,
                        kind,
                        gidx,
                        attrs,
                        dcol,
                        doff,
                        grp,
                    )
                else:
                    _class_children(
                        r,
                        m,
                        states,
                        rows_of,
                        totals,
                        class_hits,
                        codes,
                        xg,
                        parents,
                        count,
                        dense,
                        lut,
                        mean,
                        var,
                        logn,
                        kind,
                        gidx,
                        attrs,
                        dcol,
                        doff,
                        grp,
                    )
                for c in range(n_classes):
                    if states[c] >= 0:
                        group_score[grp_m, i, c] = totals[c]
                        continue
                    # The attributes with a single value add exactly 0 as
                    # children, so every component of the group gets this
                    total = log_prior_unseen[grp_m]
                    q = 0
                    for a in range(nonzero.shape[0]):
                        j = nonzero[a]
                        while q < n_dep and parents[m, q] < j:
                            q += 1
                        if q < n_dep and parents[m, q] == j:
                            continue
                        if backoff:
                            total += _log_likelihood(
                                c,
                                j,
                                r,
                                codes,
                                xg,
                                b_disc,
                                b_mean,
                                b_var,
                                b_logn,
                                kind,
                                voff,
                                gidx,
                            )
                        else:
                            total += uninf[r, j]
                    group_score[grp_m, i, c] = total
        # Pool the components in their order. The term exp(score - peak) of
        # a group is kept until the peak moves, for the other components of
        # the group: the same operands give the same term
        shape = (n_groups, hi - lo, n_classes) if reuse else (0, 0, 0)
        term = np.empty(shape)
        stamp = np.full(shape, -1, dtype=np.int64)
        moves = np.zeros((hi - lo, n_classes), dtype=np.int64)
        peak = np.full((hi - lo, n_classes), -np.inf)
        acc = np.zeros((hi - lo, n_classes))
        n_active = np.zeros(hi - lo, dtype=np.int64)
        for m in range(n_models):
            grp_m = group[m]
            for r in range(lo, hi):
                i = r - lo
                if not group_active[grp_m, i]:
                    continue
                unseen = False
                for q in range(n_dep):
                    unseen |= const[parents[m, q]] and punseen[r, parents[m, q]]
                if unseen:
                    continue
                n_active[i] += 1
                for c in range(n_classes):
                    total = group_score[grp_m, i, c]
                    if geometric:
                        acc[i, c] += total
                    elif total > peak[i, c]:
                        # Running log-sum-exp, rescaled to the largest term
                        acc[i, c] = acc[i, c] * np.exp(peak[i, c] - total) + 1.0
                        peak[i, c] = total
                        moves[i, c] += 1
                    elif total > -np.inf:
                        if not reuse:
                            acc[i, c] += np.exp(total - peak[i, c])
                        else:
                            if stamp[grp_m, i, c] != moves[i, c]:
                                term[grp_m, i, c] = np.exp(total - peak[i, c])
                                stamp[grp_m, i, c] = moves[i, c]
                            acc[i, c] += term[grp_m, i, c]
        for r in range(lo, hi):
            i = r - lo
            for c in range(n_classes):
                if n_active[i] == 0:
                    total = b_root[c]
                    for a in range(nonzero.shape[0]):
                        total += _log_likelihood(
                            c,
                            nonzero[a],
                            r,
                            codes,
                            xg,
                            b_disc,
                            b_mean,
                            b_var,
                            b_logn,
                            kind,
                            voff,
                            gidx,
                        )
                    scores[r, c] = total
                elif geometric:
                    scores[r, c] = acc[i, c] / n_active[i]
                else:
                    scores[r, c] = peak[i, c] + np.log(acc[i, c]) - np.log(n_active[i])
    return scores


# =============================================================================
# n-joins (AnJE and ALR)
# =============================================================================
#
# The cells of the n-joins live in one key space: key 0 is the class prior and
# join ``j`` owns the keys ``join_off[j] + sum_q codes[a_q] * strides[j, q]``.
# ``codes_t`` holds the attribute codes transposed, shape (n_features, n), so
# that the codes of a join are read contiguously. The table holds a row per
# cell seen in training (in the order of the sorted ``cell_keys``), then the
# empty row ``n_seen + j`` shared by the unseen cells of join ``j``. The row of
# a key comes from the dense map ``row_of_cell`` or, when the key space is too
# large for it (``row_of_cell`` is empty), from a binary search in
# ``cell_keys``.


@nb.njit(cache=True, inline="always")
def _join_key(codes_t, r, j, joins, strides, join_off):
    key = join_off[j]
    for q in range(joins.shape[1]):
        key += codes_t[joins[j, q], r] * strides[j, q]
    return key


@nb.njit(cache=True, inline="always")
def _join_row(key, j, row_of_cell, cell_keys):
    if row_of_cell.shape[0]:
        return row_of_cell[key]
    row = _find(cell_keys, 0, cell_keys.shape[0], key)
    return row if row >= 0 else cell_keys.shape[0] + j


@_Kernel
def join_cells_seen(codes_t, joins, strides, join_off, n_keys):
    """Mask of the keys that occur in ``codes_t`` (key 0 always does)."""
    seen = np.zeros(n_keys, dtype=np.bool_)
    seen[0] = True
    n = codes_t.shape[1]
    for j in nb.prange(joins.shape[0]):  # joins own disjoint keys
        if joins.shape[1] == 2:
            first, second = codes_t[joins[j, 0]], codes_t[joins[j, 1]]
            s0, s1, off = strides[j, 0], strides[j, 1], join_off[j]
            for r in range(n):
                seen[off + first[r] * s0 + second[r] * s1] = True
        else:
            for r in range(n):
                seen[_join_key(codes_t, r, j, joins, strides, join_off)] = True
    return seen


@_Kernel
def join_key_counts(codes_t, y, joins, strides, join_off, n_keys, n_classes):
    """Training count of every key and class, shape (n_keys, n_classes); key 0
    (the class prior) counts every sample."""
    n = codes_t.shape[1]
    counts = np.zeros((n_keys, n_classes), dtype=np.int32)
    for r in range(n):
        counts[0, y[r]] += 1
    flat = counts.reshape(-1)
    for j in nb.prange(joins.shape[0]):  # joins own disjoint keys
        # The constants of the join read once (pairs, the common case, have
        # their own loop): about three times faster than reading them per row
        if joins.shape[1] == 2:
            first, second = codes_t[joins[j, 0]], codes_t[joins[j, 1]]
            s0, s1, off = strides[j, 0], strides[j, 1], join_off[j]
            for r in range(n):
                flat[(off + first[r] * s0 + second[r] * s1) * n_classes + y[r]] += 1
        else:
            for r in range(n):
                key = _join_key(codes_t, r, j, joins, strides, join_off)
                flat[key * n_classes + y[r]] += 1
    return counts


@_Kernel
def join_keys(codes_t, joins, strides, join_off):
    """Key of every sample and join, shape (n, n_joins)."""
    n = codes_t.shape[1]
    keys = np.empty((n, joins.shape[0]), dtype=np.int64)
    for r in nb.prange(n):
        for j in range(joins.shape[0]):
            keys[r, j] = _join_key(codes_t, r, j, joins, strides, join_off)
    return keys


@_Kernel
def join_counts(
    codes_t, y, joins, strides, join_off, row_of_cell, cell_keys, n_classes
):
    """Training count of every table row and class, shape (n_rows, n_classes);
    row 0 (the class prior) counts every sample."""
    n = codes_t.shape[1]
    n_joins = joins.shape[0]
    counts = np.zeros((cell_keys.shape[0] + n_joins, n_classes))
    for r in range(n):
        counts[0, y[r]] += 1.0
    for j in nb.prange(n_joins):  # joins own disjoint rows
        if joins.shape[1] == 2:
            first, second = codes_t[joins[j, 0]], codes_t[joins[j, 1]]
            s0, s1, off = strides[j, 0], strides[j, 1], join_off[j]
            for r in range(n):
                key = off + first[r] * s0 + second[r] * s1
                counts[_join_row(key, j, row_of_cell, cell_keys), y[r]] += 1.0
        else:
            for r in range(n):
                key = _join_key(codes_t, r, j, joins, strides, join_off)
                counts[_join_row(key, j, row_of_cell, cell_keys), y[r]] += 1.0
    return counts


@nb.njit(cache=True, inline="always")
def _block_rows(
    lo, hi, j0, j1, codes_t, joins, strides, join_off, row_of_cell, cell_keys, rows
):
    """Table row of the joins ``j0:j1`` (axis 0, from 0) for the samples
    ``lo:hi`` (axis 1)."""
    m = hi - lo
    for j in range(j0, j1):
        keys = rows[j - j0, :m]
        keys[:] = join_off[j]
        for q in range(joins.shape[1]):
            a, s = joins[j, q], strides[j, q]
            for i in range(m):
                keys[i] += codes_t[a, lo + i] * s
        if row_of_cell.shape[0]:
            for i in range(m):
                keys[i] = row_of_cell[keys[i]]
        else:
            for i in range(m):
                keys[i] = _join_row(keys[i], j, row_of_cell, cell_keys)


@nb.njit(cache=True, inline="always")
def _block_scores(rows, off, m, table):
    """Scores ``sum_rows table[row]`` of ``m`` samples whose rows are
    ``rows[:, off:off + m]``."""
    n_classes = table.shape[1]
    scores = np.empty((m, n_classes))
    for i in range(m):
        for c in range(n_classes):
            scores[i, c] = table[0, c]
    for j in range(rows.shape[0]):
        for i in range(m):
            row = rows[j, off + i]
            for c in range(n_classes):
                scores[i, c] += table[row, c]
    return scores


@nb.njit(cache=True)
def _block_size(n_joins):
    """Samples per block of :func:`_block_rows`, so that its buffer of
    ``n_joins`` rows per sample stays within about four million entries (32 MB)."""
    return max(1, min(_BLOCK_ROWS, (1 << 22) // max(n_joins, 1)))


@_Kernel
def join_rows(codes_t, joins, strides, join_off, row_of_cell, cell_keys, by_sample):
    """Table row of every join and sample, shape (n_joins, n), or (n, n_joins)
    with ``by_sample``."""
    n = codes_t.shape[1]
    n_joins = joins.shape[0]
    shape = (n, n_joins) if by_sample else (n_joins, n)
    rows = np.empty(shape, dtype=np.int32)
    block = _block_size(n_joins)
    n_blocks = (n + block - 1) // block
    for b in nb.prange(n_blocks):
        lo, hi = b * block, min(n, (b + 1) * block)
        buf = np.empty((n_joins, block), dtype=np.int64)
        _block_rows(
            lo,
            hi,
            0,
            n_joins,
            codes_t,
            joins,
            strides,
            join_off,
            row_of_cell,
            cell_keys,
            buf,
        )
        for j in range(n_joins):
            for i in range(hi - lo):
                if by_sample:
                    rows[lo + i, j] = buf[j, i]
                else:
                    rows[j, lo + i] = buf[j, i]
    return rows


@_Kernel
def join_scores(codes_t, joins, strides, join_off, row_of_cell, cell_keys, table):
    """Score of every sample and class: the sum of the table rows of its
    cells, shape (n, n_classes). Blocks of samples go over the joins in
    chunks, so that the buffer of their rows stays within 32 MB."""
    n = codes_t.shape[1]
    n_joins = joins.shape[0]
    n_classes = table.shape[1]
    scores = np.empty((n, n_classes))
    chunk = max(1, min(n_joins, (1 << 22) // _BLOCK_ROWS))
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        buf = np.empty((chunk, _BLOCK_ROWS), dtype=np.int64)
        for i in range(lo, hi):
            for c in range(n_classes):
                scores[i, c] = table[0, c]
        for j0 in range(0, n_joins, chunk):
            j1 = min(n_joins, j0 + chunk)
            _block_rows(
                lo,
                hi,
                j0,
                j1,
                codes_t,
                joins,
                strides,
                join_off,
                row_of_cell,
                cell_keys,
                buf,
            )
            for j in range(j1 - j0):
                for i in range(hi - lo):
                    row = buf[j, i]
                    for c in range(n_classes):
                        scores[lo + i, c] += table[row, c]
    return scores


@nb.njit(cache=True)
def _binary_scores(rows, start, k, table, sums):
    """Two-class scores ``table[0] + sum_j table[rows[i, j]]`` of the ``k <= 4``
    samples ``start:start + k`` into ``sums``, each summed in join order."""
    n_joins = rows.shape[1]
    if k == 4:
        a, b, c, d = rows[start], rows[start + 1], rows[start + 2], rows[start + 3]
        a0 = b0 = c0 = d0 = table[0, 0]
        a1 = b1 = c1 = d1 = table[0, 1]
        for j in range(n_joins):
            a0 += table[a[j], 0]
            a1 += table[a[j], 1]
            b0 += table[b[j], 0]
            b1 += table[b[j], 1]
            c0 += table[c[j], 0]
            c1 += table[c[j], 1]
            d0 += table[d[j], 0]
            d1 += table[d[j], 1]
        sums[0, 0], sums[0, 1] = a0, a1
        sums[1, 0], sums[1, 1] = b0, b1
        sums[2, 0], sums[2, 1] = c0, c1
        sums[3, 0], sums[3, 1] = d0, d1
        return
    for i in range(k):
        sample = rows[start + i]
        s0, s1 = table[0, 0], table[0, 1]
        for j in range(n_joins):
            s0 += table[sample[j], 0]
            s1 += table[sample[j], 1]
        sums[i, 0], sums[i, 1] = s0, s1


@_Kernel
def join_objective(
    codes_t,
    y,
    joins,
    strides,
    join_off,
    row_of_cell,
    cell_keys,
    cached_rows,
    table,
    n_chunks,
):
    """Conditional negative log-likelihood of the scores of :func:`join_scores`
    and its derivative with respect to every table entry, shape
    (n_rows, n_classes).

    ``cached_rows`` holds the output of :func:`join_rows` for ``codes_t``
    (``by_sample`` with two classes), or is empty and the rows are computed
    block by block. The samples are split into ``n_chunks`` contiguous
    chunks with their own accumulators, summed at the end.

    Two classes, the common case, have their own loop: each sample is scored
    and its residuals spread over its rows in one visit, about three times
    faster than the loop over a run-time number of classes. With more
    classes a block of samples is processed join by join, which keeps the
    rows of the table that a join reaches in cache.
    """
    n = codes_t.shape[1]
    n_joins = joins.shape[0]
    n_rows, n_classes = table.shape
    binary = n_classes == 2
    cached = cached_rows.shape[0 if binary else 1] == n
    block = _block_size(n_joins)
    grad = np.zeros((n_chunks, n_rows, n_classes))
    nll = np.zeros(n_chunks)
    size = (n + n_chunks - 1) // n_chunks
    for t in nb.prange(n_chunks):
        buf = np.empty((n_joins, block), dtype=np.int64)
        local = np.empty((block if binary else 0, n_joins), dtype=np.int32)
        sums = np.empty((4, 2))
        g = grad[t]
        for lo in range(t * size, min(n, (t + 1) * size), block):
            hi = min(n, (t + 1) * size, lo + block)
            m = hi - lo
            if not cached:
                _block_rows(
                    lo,
                    hi,
                    0,
                    n_joins,
                    codes_t,
                    joins,
                    strides,
                    join_off,
                    row_of_cell,
                    cell_keys,
                    buf,
                )
            if binary:
                if cached:
                    rows, off = cached_rows, lo
                else:
                    for j in range(n_joins):
                        for i in range(m):
                            local[i, j] = buf[j, i]
                    rows, off = local, 0
                for i0 in range(0, m, 4):
                    k = min(4, m - i0)
                    # The scores of four samples at once: each one is still
                    # summed in the order of the joins, but the four sums
                    # do not wait on each other
                    _binary_scores(rows, off + i0, k, table, sums)
                    for i in range(i0, i0 + k):
                        sample = rows[off + i]
                        yi = y[lo + i]
                        s0, s1 = sums[i - i0, 0], sums[i - i0, 1]
                        peak = max(s0, s1)
                        log_norm = peak + np.log(np.exp(s0 - peak) + np.exp(s1 - peak))
                        nll[t] -= (s0 if yi == 0 else s1) - log_norm
                        # Softmax residuals P(c | x) - [c == y]
                        r0 = np.exp(s0 - log_norm)
                        r1 = np.exp(s1 - log_norm)
                        if yi == 0:
                            r0 -= 1.0
                        else:
                            r1 -= 1.0
                        g[0, 0] += r0
                        g[0, 1] += r1
                        for j in range(n_joins):
                            g[sample[j], 0] += r0
                            g[sample[j], 1] += r1
                continue
            if cached:
                resid = _block_scores(cached_rows, lo, m, table)
            else:
                resid = _block_scores(buf, 0, m, table)
            # Softmax residuals P(c | x) - [c == y] in place
            for i in range(m):
                peak = resid[i].max()
                total = 0.0
                for c in range(n_classes):
                    total += np.exp(resid[i, c] - peak)
                log_norm = peak + np.log(total)
                nll[t] -= resid[i, y[lo + i]] - log_norm
                for c in range(n_classes):
                    resid[i, c] = np.exp(resid[i, c] - log_norm)
                resid[i, y[lo + i]] -= 1.0
                for c in range(n_classes):
                    g[0, c] += resid[i, c]
            for j in range(n_joins):
                for i in range(m):
                    row = cached_rows[j, lo + i] if cached else buf[j, i]
                    for c in range(n_classes):
                        g[row, c] += resid[i, c]
    return nll.sum(), grad.sum(axis=0)


# =============================================================================
# Weighted AnDE: the geometric pool of exponent-weighted components
# =============================================================================


@nb.njit(cache=True)
def _pairwise_block(a, lo, n):
    """NumPy's pairwise sum of ``a[lo:lo + n]`` for ``n <= 128``: eight
    partial sums, combined as a tree, then the rest one by one."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    while i < n - n % 8:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@nb.njit(cache=True)
def _pairwise_sum(a, lo, n):
    """``np.sum(a[lo:lo + n])`` with the order of NumPy's pairwise summation,
    so that the result is the same to the bit: blocks of up to 128 items,
    and above that the sum of the two halves (the first rounded down to a
    multiple of 8). The halves are walked with a stack: a recursive function
    compiled with ``cache=True`` crashed when called from another one."""
    if n <= 128:
        return _pairwise_block(a, lo, n)
    seg_lo = np.empty(96, dtype=np.int64)
    seg_n = np.empty(96, dtype=np.int64)
    merge = np.empty(96, dtype=np.bool_)
    values = np.empty(64)
    top, n_values = 1, 0
    seg_lo[0], seg_n[0], merge[0] = lo, n, False
    while top:
        top -= 1
        start, size = seg_lo[top], seg_n[top]
        if merge[top]:
            values[n_values - 2] = values[n_values - 2] + values[n_values - 1]
            n_values -= 1
        elif size <= 128:
            values[n_values] = _pairwise_block(a, start, size)
            n_values += 1
        else:
            half = size // 2
            half -= half % 8
            # Popped in reverse: the first half, then the second, then their sum
            seg_lo[top], seg_n[top], merge[top] = start, size, True
            seg_lo[top + 1], seg_n[top + 1], merge[top + 1] = (
                start + half,
                size - half,
                False,
            )
            seg_lo[top + 2], seg_n[top + 2], merge[top + 2] = start, half, False
            top += 3
    return values[0]


@nb.njit(cache=True)
def _weighted_geometric_scores(
    index, class_step, values, n_terms, theta, intercept, active, n_classes
):
    """Pooled scores ``sum_m share[i, m] * u[i, c, m]`` of the geometric
    weighted ensemble, shape (n, n_classes).

    ``u[i, c, m]`` is the sum over the ``n_terms`` terms of the component of
    ``values[k] * theta[index[kb] + c * class_step]``, with
    ``k = ((i * n_classes + c) * n_models + m) * n_terms + t`` and
    ``kb = (i * n_models + m) * n_terms + t`` (the weights of the classes of
    a term are consecutive, or the same one), plus ``intercept`` unless it is
    empty. An empty ``values`` stands for ones, with a single term.
    ``share[i, m]`` is ``active[i, m]`` over the number of active components
    (at least 1). The operations, and their order, are those of the sparse
    product ``(operator @ theta).reshape(n, C, M) * share`` summed over the
    last axis by NumPy, without the arrays of every sample, class and
    component."""
    n, n_models = active.shape
    ones = values.shape[0] == 0
    scores = np.empty((n, n_classes))
    terms = np.empty(n_models)
    share = np.empty(n_models)
    for i in range(n):
        n_active = 0
        for m in range(n_models):
            n_active += active[i, m]
        for m in range(n_models):
            share[m] = (1.0 if active[i, m] else 0.0) / max(n_active, 1)
        for c in range(n_classes):
            row = (i * n_classes + c) * n_models
            shift = c * class_step
            if n_terms == 1:  # One weight per component, without a loop
                for m in range(n_models):
                    w = theta[np.uintp(index[i * n_models + m] + shift)]
                    u = 0.0 + (w if ones else values[row + m] * w)
                    if intercept.shape[0]:
                        u += intercept[i, c, m]
                    terms[m] = u * share[m]
            else:
                for m in range(n_models):
                    start = np.uintp((row + m) * n_terms)
                    base = np.uintp((i * n_models + m) * n_terms)
                    u = 0.0
                    # Unsigned indices: no check for negative ones at every read
                    for t in range(np.uintp(n_terms)):
                        u += (
                            values[start + t] * theta[np.uintp(index[base + t] + shift)]
                        )
                    if intercept.shape[0]:
                        u += intercept[i, c, m]
                    terms[m] = u * share[m]
            scores[i, c] = 0.0 + _pairwise_sum(terms, 0, n_models)
    return scores


@nb.njit(cache=True)
def _weighted_geometric_gradient(
    index, class_step, values, n_terms, dscores, active, n_weights
):
    """Gradient of the loss with respect to ``theta`` given its derivative
    ``dscores`` with respect to the scores of
    :func:`_weighted_geometric_scores`: the transposed sparse product
    ``operator.T @ (dscores[:, :, None] * share).ravel()``, accumulated in the
    order of its rows, as SciPy does."""
    n, n_models = active.shape
    n_classes = dscores.shape[1]
    ones = values.shape[0] == 0
    grad = np.zeros(n_weights)
    share = np.empty(n_models)
    for i in range(n):
        n_active = 0
        for m in range(n_models):
            n_active += active[i, m]
        for m in range(n_models):
            share[m] = (1.0 if active[i, m] else 0.0) / max(n_active, 1)
        for c in range(n_classes):
            row = (i * n_classes + c) * n_models
            shift = c * class_step
            if n_terms == 1:
                for m in range(n_models):
                    du = dscores[i, c] * share[m]
                    j = np.uintp(index[i * n_models + m] + shift)
                    grad[j] += du if ones else values[row + m] * du
                continue
            for m in range(n_models):
                start = np.uintp((row + m) * n_terms)
                base = np.uintp((i * n_models + m) * n_terms)
                du = dscores[i, c] * share[m]
                for t in range(np.uintp(n_terms)):
                    grad[np.uintp(index[base + t] + shift)] += values[start + t] * du
    return grad


@nb.njit(cache=True)
def _component_terms(
    i, index, class_step, values, n_terms, theta, intercept, n_classes, posterior, u
):
    """Component scores ``u[c, m]`` of sample ``i``, as in
    :func:`_weighted_geometric_scores`; with ``posterior``, each component's
    scores minus their log-sum-exp over the classes."""
    n_models = u.shape[1]
    ones = values.shape[0] == 0
    for c in range(n_classes):
        row = (i * n_classes + c) * n_models
        shift = c * class_step
        for m in range(n_models):
            start = np.uintp((row + m) * n_terms)
            base = np.uintp((i * n_models + m) * n_terms)
            v = 0.0
            for t in range(np.uintp(n_terms)):
                w = theta[np.uintp(index[base + t] + shift)]
                v += w if ones else values[start + t] * w
            if intercept.shape[0]:
                v += intercept[i, c, m]
            u[c, m] = v
    if posterior:
        column = np.empty(n_classes)
        work = np.empty(n_classes)
        for m in range(n_models):
            for c in range(n_classes):
                column[c] = u[c, m]
            # NumPy reduces the class axis, not the last one, in order
            norm = _log_sum_exp(column, work, False)
            for c in range(n_classes):
                u[c, m] -= norm


@nb.njit(cache=True)
def _log_sum_exp(v, work, pairwise):
    """``log(sum(exp(v)))`` as SciPy's ``logsumexp`` computes it (1.15 on):
    the largest entry ``peak``, found ``k`` times, plus ``log(k)`` plus
    ``log1p`` of the other entries' ``exp(x - peak)`` summed and divided by
    ``k``; the sum in NumPy's pairwise order (``pairwise``, a reduction over
    the last axis) or in order (over another axis). ``work`` holds the terms."""
    n = v.shape[0]
    peak = v[0]
    for j in range(1, n):
        peak = max(peak, v[j])
    count = 0
    for j in range(n):
        count += v[j] == peak
    shift = peak if -np.inf < peak < np.inf else 0.0
    for j in range(n):
        work[j] = 0.0 if v[j] == peak else np.exp(v[j] - shift)
    if pairwise:
        total = 0.0 + _pairwise_sum(work, 0, n)
    else:
        total = 0.0
        for j in range(n):
            total += work[j]
    if total != 0.0:
        total /= count
    return np.log1p(total) + np.log(count) + peak


@nb.njit(cache=True)
def _weighted_arithmetic_scores(
    index,
    class_step,
    values,
    n_terms,
    theta,
    intercept,
    active,
    n_classes,
    posterior,
):
    """Pooled scores ``log sum_m exp(u[i, c, m])`` over the active components
    of the arithmetic weighted ensemble, shape (n, n_classes), 0 for a sample
    with no active component, and the component scores ``u``, shape
    (n, n_classes, n_models), which the gradient reuses. ``u`` is that of
    :func:`_weighted_geometric_scores`; with ``posterior`` every component's
    scores are normalized over the classes first."""
    n, n_models = active.shape
    scores = np.empty((n, n_classes))
    terms = np.empty((n, n_classes, n_models))
    masked = np.empty(n_models)
    work = np.empty(n_models)
    for i in range(n):
        u = terms[i]
        _component_terms(
            i,
            index,
            class_step,
            values,
            n_terms,
            theta,
            intercept,
            n_classes,
            posterior,
            u,
        )
        n_active = 0
        for m in range(n_models):
            n_active += active[i, m]
        for c in range(n_classes):
            if n_active == 0:
                scores[i, c] = 0.0
                continue
            for m in range(n_models):
                masked[m] = u[c, m] if active[i, m] else -np.inf
            scores[i, c] = _log_sum_exp(masked, work, True)
    return scores, terms


@nb.njit(cache=True)
def _weighted_arithmetic_gradient(
    index,
    class_step,
    values,
    n_terms,
    terms,
    active,
    scores,
    dscores,
    posterior,
    n_weights,
):
    """Gradient of the loss with respect to ``theta`` for the scores and the
    component scores ``terms`` of :func:`_weighted_arithmetic_scores`, given
    the derivative ``dscores`` with respect to the scores: every component
    score gets ``dscores`` times its responsibility ``exp(u - score)``
    (through the normalization over the classes with ``posterior``), spread
    over the weights of its terms."""
    n, n_models = active.shape
    n_classes = dscores.shape[1]
    ones = values.shape[0] == 0
    grad = np.zeros(n_weights)
    du = np.empty((n_classes, n_models))
    for i in range(n):
        u = terms[i]
        for c in range(n_classes):
            for m in range(n_models):
                resp = 0.0
                if active[i, m]:
                    resp = np.exp(u[c, m] - scores[i, c])
                    if resp != resp:
                        resp = 0.0
                du[c, m] = dscores[i, c] * resp
        if posterior:
            # Through log_q = u - logsumexp_c u, with q = exp(log_q) = exp(u)
            for m in range(n_models):
                total = 0.0
                for c in range(n_classes):
                    total += du[c, m]
                for c in range(n_classes):
                    du[c, m] -= np.exp(u[c, m]) * total
        for c in range(n_classes):
            row = (i * n_classes + c) * n_models
            shift = c * class_step
            for m in range(n_models):
                start = np.uintp((row + m) * n_terms)
                base = np.uintp((i * n_models + m) * n_terms)
                for t in range(np.uintp(n_terms)):
                    j = np.uintp(index[base + t] + shift)
                    grad[j] += du[c, m] if ones else values[start + t] * du[c, m]
    return grad


# =============================================================================
# Attribute-conditional networks (TAN and KDB)
# =============================================================================
#
# Factor ``m`` is ``P(x_j | y, x_pa)`` of child ``j = children[m]``. Its
# augmented states are stored as those of an SPnDE; see ConditionalTables.


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
    """The fitted conditionals of a TAN or KDB network, in the order of the
    parameters of :func:`conditional_scores`.

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
def conditional_scores(
    codes,
    xg,
    uninf,
    pcodes,
    punseen,
    parents,
    strides,
    n_parents,
    children,
    n_combos,
    state_off,
    state_keys,
    disc_off,
    disc,
    cell_off,
    cell_keys,
    cell_unseen,
    mean,
    var,
    logn,
    kind,
    gidx,
    cards,
    n_classes,
    per_factor,
):
    """Sum over the factors of ``log P(x_j | y, x_pa)``, shape
    (n, n_classes, 1). With ``per_factor`` the factors are kept apart instead,
    factor ``m`` at ``[:, :, 1 + m]``, and ``[:, :, 0]`` is left at 0 for the
    class log-prior, which the caller adds.

    A state never seen in training, or a parent value never seen, gives the
    uninformed estimate ``uninf[r, j]``.
    """
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


# =============================================================================
# Label encoding of integer attributes
# =============================================================================


@nb.njit(cache=True)
def _column_ranges(X, columns):
    """Smallest and largest value of every column in ``columns``."""
    k = columns.shape[0]
    lo = np.empty(k, dtype=np.int64)
    hi = np.empty(k, dtype=np.int64)
    for c in range(k):
        lo[c] = hi[c] = X[0, columns[c]]
    for r in range(1, X.shape[0]):
        for c in range(k):
            v = X[r, columns[c]]
            lo[c] = min(lo[c], v)
            hi[c] = max(hi[c], v)
    return lo, hi


@nb.njit(cache=True)
def _integer_codes(X, columns, lo, table_off):
    """Rank of every value of the integer columns ``columns`` among the
    distinct values of its column, int32 of shape (n, len(columns)), and the
    table
    from value to rank: column ``c`` owns the entries
    ``table_off[c]:table_off[c + 1]`` for the values from ``lo[c]`` up,
    -1 for the values absent from it."""
    n = X.shape[0]
    k = columns.shape[0]
    table = np.full(table_off[k], -1, dtype=np.int64)
    for r in range(n):
        for c in range(k):
            table[table_off[c] + X[r, columns[c]] - lo[c]] = 0
    for c in range(k):
        rank = 0
        for v in range(table_off[c], table_off[c + 1]):
            if table[v] == 0:
                table[v] = rank
                rank += 1
    codes = np.empty((n, k), dtype=np.int32)
    for r in range(n):
        for c in range(k):
            codes[r, c] = table[table_off[c] + X[r, columns[c]] - lo[c]]
    return codes, table


@nb.njit(cache=True)
def _categorical_counts(X, y, offsets, cards, size):
    """Count of every (feature, class, value) cell of a categorical Naive
    Bayes: feature ``j`` owns the cells ``offsets[j] + y * cards[j] + v``."""
    flat = np.zeros(size)
    for r in range(X.shape[0]):
        for j in range(X.shape[1]):
            flat[offsets[j] + y[r] * cards[j] + X[r, j]] += 1.0
    return flat
