"""Compiled kernels that count and evaluate the SPnDEs of an ensemble.

Every SPnDE is a Naive Bayes model on the augmented class ``Y* = (y, x_p)``.
Its parameters are stored per augmented state observed in training, in flat
tables shared by all the components of the ensemble:

- ``state_keys[s]``: the key ``y * n_combos + parent_key`` of state ``s``; the
  states of component ``m`` are ``state_off[m]:state_off[m + 1]``, sorted.
- ``disc[s, voff[j] + v]``: ``log P(x_j = v | s)`` of a discrete attribute
  ``j`` (categorical, or Bernoulli with values 0 and 1), with ``-inf``
  replaced by ``LOG_ZERO``.
- ``mean[s, g]`` and ``var[s, g]``: the Gaussian of continuous attribute ``g``.

``kind[j]`` gives the type of attribute ``j`` (``IGNORED``, ``DISCRETE`` or
``GAUSSIAN``), ``voff[j]`` its offset in the rows of ``disc`` and ``gidx[j]``
its column in ``mean`` and ``var``. A discrete code of -1 marks a value never
seen in training, which contributes nothing.
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
def _augmented_keys(y, pcodes, parents, strides, n_combos):
    """Key ``y * n_combos + parent_key`` of every training row."""
    n = y.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for r in range(n):
        key = 0
        for q in range(parents.shape[0]):
            key += pcodes[r, parents[q]] * strides[q]
        keys[r] = y[r] * n_combos + key
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
    states = np.unique(keys)
    return states, np.searchsorted(states, keys)


@_Kernel
def count_states(y, pcodes, parents, strides, n_combos, n_classes):
    """Number of augmented states observed in training by each component."""
    n_models = parents.shape[0]
    n_states = np.empty(n_models, dtype=np.int64)
    for m in nb.prange(n_models):
        keys = _augmented_keys(y, pcodes, parents[m], strides[m], n_combos[m])
        n_states[m] = _state_index(keys, n_classes * n_combos[m])[0].shape[0]
    return n_states


@_Kernel
def count_tables(
    y,
    pcodes,
    codes,
    xg,
    parents,
    strides,
    children,
    n_combos,
    n_classes,
    state_off,
    kind,
    voff,
    gidx,
    n_values,
):
    """Sufficient statistics of every component.

    Returns the state keys, the number of training rows of every state, the
    counts of every discrete value per state, and the mean and the (biased)
    variance of every Gaussian attribute per state. The rows of the parents
    of a component are left at zero.
    """
    n = codes.shape[0]
    n_models = parents.shape[0]
    n_total = state_off[n_models]
    n_gauss = xg.shape[1]
    keys_out = np.empty(n_total, dtype=np.int64)
    count = np.zeros(n_total)
    disc = np.zeros((n_total, n_values))
    mean = np.zeros((n_total, n_gauss))
    var = np.zeros((n_total, n_gauss))
    for m in nb.prange(n_models):
        keys = _augmented_keys(y, pcodes, parents[m], strides[m], n_combos[m])
        states, index = _state_index(keys, n_classes * n_combos[m])
        o = state_off[m]
        keys_out[o : o + states.shape[0]] = states
        for r in range(n):
            s = o + index[r]
            count[s] += 1.0
            for j in children[m]:
                if kind[j] == DISCRETE:
                    disc[s, voff[j] + codes[r, j]] += 1.0
                elif kind[j] == GAUSSIAN:
                    mean[s, gidx[j]] += xg[r, gidx[j]]
        if n_gauss == 0:
            continue
        for s in range(o, o + states.shape[0]):
            for g in range(n_gauss):
                mean[s, g] /= count[s]
        for r in range(n):
            s = o + index[r]
            for j in children[m]:
                if kind[j] == GAUSSIAN:
                    diff = xg[r, gidx[j]] - mean[s, gidx[j]]
                    var[s, gidx[j]] += diff * diff
        for s in range(o, o + states.shape[0]):
            for g in range(n_gauss):
                var[s, g] /= count[s]
    return keys_out, count, disc, mean, var


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
def _log_likelihood(s, j, r, codes, xg, disc, mean, var, kind, voff, gidx):
    """``log P(x_rj | state s)``, clamped as ``np.nan_to_num`` (the discrete
    tables are clamped when they are built)."""
    if kind[j] == DISCRETE:
        code = codes[r, j]
        if code < 0:
            return 0.0
        return disc[s, voff[j] + code]
    if kind[j] == GAUSSIAN:
        g = gidx[j]
        v = var[s, g]
        diff = xg[r, g] - mean[s, g]
        return _clamp(-0.5 * (diff * diff / v + np.log(2.0 * np.pi * v)))
    return 0.0


# The same function called rather than inlined, on the back-off paths that
# few samples take: it keeps the evaluation kernels faster to compile
_log_likelihood_call = nb.njit(cache=True)(_log_likelihood.py_func)


@nb.njit(cache=True, inline="always")
def _parent_state(r, m, pcodes, punseen, parents, strides, pk_off, pk_keys, pk_count):
    """Parent key of component ``m`` for row ``r`` and the training frequency
    of that parent instantiation (0 when a parent value was never seen)."""
    key = 0
    unseen = False
    for q in range(parents.shape[1]):
        p = parents[m, q]
        key += pcodes[r, p] * strides[m, q]
        unseen |= punseen[r, p]
    pos = _find(pk_keys, pk_off[m], pk_off[m + 1], key)
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
    children,
    n_combos,
    state_off,
    state_keys,
    root,
    disc,
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
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        for m in range(n_models):
            for r in range(lo, hi):
                key, freq = _parent_state(
                    r, m, pcodes, punseen, parents, strides, pk_off, pk_keys, pk_count
                )
                act = freq >= min_count or n_dep == 0
                active[r, m] = act
                for c in range(n_classes):
                    s = _find(
                        state_keys,
                        state_off[m],
                        state_off[m + 1],
                        c * n_combos[m] + key,
                    )
                    if s < 0 and not act:
                        continue
                    total = root[s] if s >= 0 else log_prior_unseen[m]
                    out[r, c, m, 0] = total
                    for t in range(children.shape[1]):
                        j = children[m, t]
                        if s >= 0:
                            ll = _log_likelihood(
                                s, j, r, codes, xg, disc, mean, var, kind, voff, gidx
                            )
                        elif backoff:
                            ll = _log_likelihood_call(
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
                            out[r, c, m, 1 + t] = ll
                        else:
                            total += ll
                    if not per_factor:
                        out[r, c, m, 0] = total
        for r in range(lo, hi):
            if active[r].any():
                continue
            for c in range(n_classes):
                total = b_root[c]
                for j in range(n_features):
                    total += _log_likelihood_call(
                        c, j, r, codes, xg, b_disc, b_mean, b_var, kind, voff, gidx
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
    children,
    n_combos,
    state_off,
    state_keys,
    root,
    disc,
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
    min_count,
    backoff,
    geometric,
):
    """Pooled score of every sample and class, shape (n, n_classes).

    The arithmetic pool is ``log mean_m P_m(y, x)`` and the geometric one
    ``mean_m log P_m(y, x)``, over the active components; samples with no
    active component get the Naive Bayes joint. Equals pooling the output of
    :func:`component_scores` without materialising it.
    """
    n, n_features = codes.shape
    n_models, n_dep = parents.shape
    n_classes = b_root.shape[0]
    scores = np.empty((n, n_classes))
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        peak = np.full((hi - lo, n_classes), -np.inf)
        acc = np.zeros((hi - lo, n_classes))
        n_active = np.zeros(hi - lo, dtype=np.int64)
        for m in range(n_models):
            for r in range(lo, hi):
                key, freq = _parent_state(
                    r, m, pcodes, punseen, parents, strides, pk_off, pk_keys, pk_count
                )
                if not (freq >= min_count or n_dep == 0):
                    continue
                i = r - lo
                n_active[i] += 1
                for c in range(n_classes):
                    s = _find(
                        state_keys,
                        state_off[m],
                        state_off[m + 1],
                        c * n_combos[m] + key,
                    )
                    total = root[s] if s >= 0 else log_prior_unseen[m]
                    for j in children[m]:
                        if s >= 0:
                            total += _log_likelihood(
                                s, j, r, codes, xg, disc, mean, var, kind, voff, gidx
                            )
                        elif backoff:
                            total += _log_likelihood_call(
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
                    if geometric:
                        acc[i, c] += total
                    elif total > peak[i, c]:
                        # Running log-sum-exp, rescaled to the largest term
                        acc[i, c] = acc[i, c] * np.exp(peak[i, c] - total) + 1.0
                        peak[i, c] = total
                    elif total > -np.inf:
                        acc[i, c] += np.exp(total - peak[i, c])
        for r in range(lo, hi):
            i = r - lo
            for c in range(n_classes):
                if n_active[i] == 0:
                    total = b_root[c]
                    for j in range(n_features):
                        total += _log_likelihood_call(
                            c, j, r, codes, xg, b_disc, b_mean, b_var, kind, voff, gidx
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
        for r in range(n):
            seen[_join_key(codes_t, r, j, joins, strides, join_off)] = True
    return seen


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
        for r in range(n):
            key = _join_key(codes_t, r, j, joins, strides, join_off)
            counts[_join_row(key, j, row_of_cell, cell_keys), y[r]] += 1.0
    return counts


@nb.njit(cache=True, inline="always")
def _block_rows(
    lo, hi, codes_t, joins, strides, join_off, row_of_cell, cell_keys, rows
):
    """Table row of every join (axis 0) for the samples ``lo:hi`` (axis 1)."""
    m = hi - lo
    for j in range(joins.shape[0]):
        keys = rows[j, :m]
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


@_Kernel
def join_rows(codes_t, joins, strides, join_off, row_of_cell, cell_keys):
    """Table row of every join and sample, shape (n_joins, n)."""
    n = codes_t.shape[1]
    rows = np.empty((joins.shape[0], n), dtype=np.int64)
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        buf = np.empty((joins.shape[0], _BLOCK_ROWS), dtype=np.int64)
        _block_rows(
            lo, hi, codes_t, joins, strides, join_off, row_of_cell, cell_keys, buf
        )
        rows[:, lo:hi] = buf[:, : hi - lo]
    return rows


@_Kernel
def join_scores(codes_t, joins, strides, join_off, row_of_cell, cell_keys, table):
    """Score of every sample and class: the sum of the table rows of its
    cells, shape (n, n_classes)."""
    n = codes_t.shape[1]
    scores = np.empty((n, table.shape[1]))
    n_blocks = (n + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    for b in nb.prange(n_blocks):
        lo, hi = b * _BLOCK_ROWS, min(n, (b + 1) * _BLOCK_ROWS)
        buf = np.empty((joins.shape[0], _BLOCK_ROWS), dtype=np.int64)
        _block_rows(
            lo, hi, codes_t, joins, strides, join_off, row_of_cell, cell_keys, buf
        )
        scores[lo:hi] = _block_scores(buf, 0, hi - lo, table)
    return scores


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

    ``cached_rows`` holds the output of :func:`join_rows` for ``codes_t``, or
    is empty and the rows are computed block by block. The samples are split
    into ``n_chunks`` contiguous chunks with their own accumulators, summed at
    the end.
    """
    n = codes_t.shape[1]
    n_rows, n_classes = table.shape
    cached = cached_rows.shape[1] == n
    grad = np.zeros((n_chunks, n_rows, n_classes))
    nll = np.zeros(n_chunks)
    size = (n + n_chunks - 1) // n_chunks
    for t in nb.prange(n_chunks):
        buf = np.empty((joins.shape[0], _BLOCK_ROWS), dtype=np.int64)
        for lo in range(t * size, min(n, (t + 1) * size), _BLOCK_ROWS):
            hi = min(n, (t + 1) * size, lo + _BLOCK_ROWS)
            m = hi - lo
            if cached:
                rows, off = cached_rows, lo
            else:
                _block_rows(
                    lo,
                    hi,
                    codes_t,
                    joins,
                    strides,
                    join_off,
                    row_of_cell,
                    cell_keys,
                    buf,
                )
                rows, off = buf, 0
            resid = _block_scores(rows, off, m, table)
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
                    grad[t, 0, c] += resid[i, c]
            for j in range(rows.shape[0]):
                for i in range(m):
                    row = rows[j, off + i]
                    for c in range(n_classes):
                        grad[t, row, c] += resid[i, c]
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
    pcodes,
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
        keys = _augmented_keys(y, pcodes, parents[m], strides[m], n_combos[m])
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
