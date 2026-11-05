"""Compiled kernels that count and evaluate the SPnDEs of an ensemble.

Every SPnDE is a Naive Bayes model on the augmented class ``Y* = (y, x_p)``.
Its parameters are stored per augmented state observed in training, in flat
tables shared by all the components of the ensemble. The components that
differ only in which constant attributes (``const``: one value as parents, no
information as children) are among their parents have the same parameters,
stored once for their group ``g = group[m]``:

- ``state_keys[s]``: the key ``y * n_combos + parent_key`` of state ``s``; the
  states of group ``g`` are ``state_off[g]:state_off[g + 1]``, sorted.
  ``count[s]`` is its number of training rows.
- ``log P(x_j = v | s)`` of a discrete attribute ``j`` (categorical, or
  Bernoulli with values 0 and 1), with ``-inf`` replaced by ``LOG_ZERO``. A
  dense state (more than ``_MAX_LIGHT`` training rows) keeps it in its row
  ``drow[s]`` of ``dense``, at column ``doff[j] + v``. A light state keeps
  only its training rows, ``rows[loff[s]:loff[s] + count[s]]``: the number
  ``k`` of them with ``x_j = v`` (from the training codes ``tcodes``, column
  ``dcol[j]``) gives the estimate ``lut[count[s], grp[j], k]``, where
  ``grp[j]`` indexes the cardinality of ``j``. Most states of high-order or
  high-cardinality ensembles have a few rows, and a dense row per state
  would cost far more memory.
- ``mean[s, g]`` and ``var[s, g]``: the Gaussian of continuous attribute ``g``.

``kind[j]`` gives the type of attribute ``j`` (``IGNORED``, ``DISCRETE`` or
``GAUSSIAN``) and ``gidx[j]`` its column in ``mean`` and ``var``; ``attrs``
lists the attributes that can contribute as children (the Gaussian ones and
the discrete ones with more than one value; a single value has probability
1). The class-level back-off tables keep every discrete value, at
``voff[j] + v``. A discrete code of -1 marks a value never seen in training,
which contributes nothing.
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


@_Kernel
def count_states(y, pcodes_t, parents, strides, n_combos, n_classes):
    """Number of augmented states observed in training by each component."""
    n_models = parents.shape[0]
    n_states = np.empty(n_models, dtype=np.int64)
    for m in nb.prange(n_models):
        keys = _augmented_keys(y, pcodes_t, parents[m], strides[m], n_combos[m])
        n_states[m] = _state_index(keys, n_classes * n_combos[m])[0].shape[0]
    return n_states


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
    by_triples,
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
    zero, and so are those of the discrete children that
    :func:`count_triples` counts: when ``by_triples[m]``, the children ``j``
    of the parents ``p < q`` whose triple has at most ``max_cells`` cells
    (``n_classes * pcards[p] * pcards[q] * pcards[j]``).
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
            if by_triples[m] and (
                n_classes * pcards[parents[m, 0]] * pcards[parents[m, 1]] * pcards[j]
                <= max_cells
            ):
                continue  # Counted by count_triples
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
            counts = cells[:size]
            counts[:] = 0
            for r in range(n):
                counts[base[r] * cc + xc[r]] += 1
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
def _gaussian_ll(x, mean, var):
    """Gaussian log-density, clamped as ``np.nan_to_num``."""
    diff = x - mean
    return _clamp(-0.5 * (diff * diff / var + np.log(2.0 * np.pi * var)))


@nb.njit(cache=True)
def _log_likelihood(s, j, r, codes, xg, disc, mean, var, kind, voff, gidx):
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
        return _gaussian_ll(xg[r, g], mean[s, g], var[s, g])
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


@_Kernel
def component_scores(
    codes,
    xg,
    uninf,
    pcodes,
    punseen,
    parents,
    strides,
    n_combos,
    group,
    rep,
    const,
    state_off,
    state_keys,
    root,
    count,
    drow,
    loff,
    dense,
    rows,
    tcodes,
    lut,
    mean,
    var,
    pk_off,
    pk_keys,
    pk_count,
    log_prior_unseen,
    b_root,
    b_disc,
    b_mean,
    b_var,
    kind,
    voff,
    gidx,
    attrs,
    dcol,
    doff,
    grp,
    nonzero,
    min_count,
    backoff,
    per_factor,
):
    """Log-joint ``log P_m(y, x)`` of every sample, class and component.

    Returns ``out`` of shape (n, n_classes, n_models), or with
    ``per_factor`` of shape (n, n_classes, n_models, 1 + n_children) holding
    the root and child factors; ``active`` of shape (n, n_models); and the
    Naive Bayes joint of the samples with no active component (zero for the
    others).
    """
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
                                ll = _gaussian_ll(xg[r, g], mean[s, g], var[s, g])
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
                        kind,
                        voff,
                        gidx,
                    )
                nb_jll[r, c] = total
    return out, active, nb_jll


@_Kernel
def pooled_scores(
    codes,
    xg,
    uninf,
    pcodes,
    punseen,
    parents,
    strides,
    n_combos,
    group,
    rep,
    const,
    state_off,
    state_keys,
    root,
    count,
    drow,
    loff,
    dense,
    rows,
    tcodes,
    lut,
    mean,
    var,
    pk_off,
    pk_keys,
    pk_count,
    log_prior_unseen,
    b_root,
    b_disc,
    b_mean,
    b_var,
    kind,
    voff,
    gidx,
    attrs,
    dcol,
    doff,
    grp,
    nonzero,
    min_count,
    backoff,
    geometric,
):
    """Pooled score of every sample and class, shape (n, n_classes).

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
                for c in range(n_classes):
                    s = _find(
                        state_keys,
                        state_off[grp_m],
                        state_off[grp_m + 1],
                        c * n_combos[m] + key,
                    )
                    q = 0
                    if s >= 0:
                        # As in component_scores
                        if drow[s] < 0:
                            _count_hits(
                                s, count, loff, rows, tcodes, sample_codes[i], hits
                            )
                        total = root[s]
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
                                    total += dense[drow[s], doff[j] + code]
                                else:
                                    total += lut[int(count[s]), grp[j], hits[dcol[j]]]
                            else:
                                g = gidx[j]
                                total += _gaussian_ll(xg[r, g], mean[s, g], var[s, g])
                    else:
                        # The attributes with a single value add exactly 0 as
                        # children, so every component of the group gets this
                        total = log_prior_unseen[grp_m]
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
# Attribute-conditional networks (TAN and KDB)
# =============================================================================
#
# Factor ``m`` is ``P(x_j | y, x_pa)`` of child ``j = children[m]`` with the
# ``n_parents[m]`` parents in ``parents[m]`` (the rest of the row is padding
# with stride 0). Its augmented states are stored as those of an SPnDE
# (``state_off``, ``state_keys``); a discrete child keeps
# ``disc[disc_off[m] + t * cards[j] + v]`` for its local state ``t``, and a
# Gaussian child ``mean[s]`` and ``var[s]``.


@_Kernel
def conditional_mutual_info(codes_t, y, cards, n_classes):
    """Empirical ``I(A_i; A_j | C)`` of every pair of attributes, in nats.

    ``codes_t`` holds the attribute codes transposed, shape (n_features, n).
    The (c, a) counts of every attribute are computed once; every pair counts
    its joint table and sums ``n_cab log(n_cab n_c / (n_ca n_cb))`` over its
    non-empty cells, which is exactly 0 for a constant attribute.
    """
    d, n = codes_t.shape
    max_card = 1
    for j in range(d):
        max_card = max(max_card, cards[j])
    n_c = np.zeros(n_classes)
    for r in range(n):
        n_c[y[r]] += 1.0
    marg = np.zeros((d, n_classes, max_card))
    for j in nb.prange(d):
        for r in range(n):
            marg[j, y[r], codes_t[j, r]] += 1.0
    cmi = np.zeros((d, d))
    for i in nb.prange(d):
        ci = cards[i]
        key = np.empty(n, dtype=np.int64)
        for r in range(n):
            key[r] = (y[r] * ci + codes_t[i, r]) * max_card
        table = np.zeros(n_classes * ci * max_card)
        for j in range(i + 1, d):
            cj = cards[j]
            table[:] = 0.0
            for r in range(n):
                table[key[r] + codes_t[j, r]] += 1.0
            total = 0.0
            for c in range(n_classes):
                for a in range(ci):
                    row = (c * ci + a) * max_card
                    for b in range(cj):
                        n_cab = table[row + b]
                        if n_cab > 0.0:
                            ratio = n_cab * n_c[c] / (marg[i, c, a] * marg[j, c, b])
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
    kind,
    gidx,
    cards,
):
    """Sufficient statistics of every factor: the state keys, the training
    rows of every state, the value counts of a discrete child and the mean
    and (biased) variance of a Gaussian one."""
    n = y.shape[0]
    n_factors = parents.shape[0]
    n_total = state_off[n_factors]
    keys_out = np.empty(n_total, dtype=np.int64)
    count = np.zeros(n_total)
    disc = np.zeros(disc_off[n_factors])
    mean = np.zeros(n_total)
    var = np.zeros(n_total)
    for m in nb.prange(n_factors):
        keys = _augmented_keys(y, pcodes_t, parents[m], strides[m], n_combos[m])
        states, index = _state_index(keys, n_classes * n_combos[m])
        o = state_off[m]
        keys_out[o : o + states.shape[0]] = states
        j = children[m]
        for r in range(n):
            s = o + index[r]
            count[s] += 1.0
            if kind[j] == DISCRETE:
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
    return keys_out, count, disc, mean, var


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
    mean,
    var,
    kind,
    gidx,
    cards,
    n_classes,
):
    """Sum over the factors of ``log P(x_j | y, x_pa)``, shape (n, n_classes).

    A state never seen in training, or a parent value never seen, gives the
    uninformed estimate ``uninf[r, j]``.
    """
    n = codes.shape[0]
    n_factors = parents.shape[0]
    out = np.zeros((n, n_classes))
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        for m in range(n_factors):
            j = children[m]
            o = state_off[m]
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
                        out[r, c] += uninf[r, j]
                    elif kind[j] == DISCRETE:
                        code = codes[r, j]
                        if code >= 0:
                            out[r, c] += disc[disc_off[m] + (s - o) * cards[j] + code]
                    elif kind[j] == GAUSSIAN:
                        v = var[s]
                        diff = xg[r, gidx[j]] - mean[s]
                        out[r, c] += _clamp(
                            -0.5 * (diff * diff / v + np.log(2.0 * np.pi * v))
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
