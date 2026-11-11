"""Compiled kernels of the n-joins (AnJE and ALR).

The cells of the n-joins live in one key space: key 0 is the class prior and
join ``j`` owns the keys ``join_off[j] + sum_q codes[a_q] * strides[j, q]``.
``codes_t`` holds the attribute codes transposed, shape (n_features, n), so
that the codes of a join are read contiguously. The table holds a row per
cell seen in training (in the order of the sorted ``cell_keys``), then the
empty row ``n_seen + j`` shared by the unseen cells of join ``j``. The row of
a key comes from the dense map ``row_of_cell`` or, when the key space is too
large for it (``row_of_cell`` is empty), from a binary search in
``cell_keys``.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np

from ._runtime import _BLOCK_ROWS, _find, _Kernel


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
