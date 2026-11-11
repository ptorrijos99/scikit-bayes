"""Label encoding of integer attributes, and the per-class sufficient
statistics of Naive Bayes.

The profiles, ranges and codes of the input columns that the attribute
encoder of :mod:`skbn._core` reads, and the Gaussian moments and categorical
counts of :class:`~skbn.MixedNB`.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _column_ranges(X, columns):
    """Smallest and largest value of every column in ``columns`` of an
    integer ``X``."""
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
def _column_profiles(X, columns):
    """For every column in ``columns``: its smallest and largest value (as
    floats), whether all its values are integers and how many distinct values
    it has, counted up to three. One pass over the rows gives what typing an
    attribute needs, without sorting it."""
    k = columns.shape[0]
    lo = np.empty(k)
    hi = np.empty(k)
    whole = np.ones(k, dtype=np.bool_)
    first = np.empty(k)
    second = np.empty(k)
    distinct = np.ones(k, dtype=np.int64)
    for c in range(k):
        lo[c] = hi[c] = first[c] = second[c] = X[0, columns[c]]
    for r in range(X.shape[0]):
        for c in range(k):
            v = X[r, columns[c]]
            lo[c] = min(lo[c], v)
            hi[c] = max(hi[c], v)
            whole[c] &= v == np.floor(v)
            if distinct[c] == 1:
                if v != first[c]:
                    second[c] = v
                    distinct[c] = 2
            elif distinct[c] == 2 and v != first[c] and v != second[c]:
                distinct[c] = 3
    return lo, hi, whole, distinct


@nb.njit(cache=True, error_model="numpy")
def _class_moments(X, y, n_classes):
    """Mean and variance of every column of ``X`` within each class, each of
    shape (n_classes, n_features), and the class counts. The sums run over
    the rows in order, as ``np.bincount`` with weights does, without the
    array of squared deviations."""
    n, d = X.shape
    count = np.zeros(n_classes)
    mean = np.zeros((n_classes, d))
    var = np.zeros((n_classes, d))
    for r in range(n):
        c = y[r]
        count[c] += 1.0
        for j in range(d):
            mean[c, j] += X[r, j]
    for c in range(n_classes):
        for j in range(d):
            mean[c, j] /= count[c]
    for r in range(n):
        c = y[r]
        for j in range(d):
            diff = X[r, j] - mean[c, j]
            var[c, j] += diff * diff
    for c in range(n_classes):
        for j in range(d):
            var[c, j] /= count[c]
    return count, mean, var


@nb.njit(cache=True)
def _integer_codes(X, columns, lo, table_off):
    """Rank of every value of the columns ``columns``, which hold integers
    (in an integer or a floating ``X``), among the distinct values of its
    column, int32 of shape (n, len(columns)), and the table from value to
    rank: column ``c`` owns the entries ``table_off[c]:table_off[c + 1]`` for
    the values from ``lo[c]`` up, -1 for the values absent from it."""
    n = X.shape[0]
    k = columns.shape[0]
    table = np.full(table_off[k], -1, dtype=np.int64)
    for r in range(n):
        for c in range(k):
            table[table_off[c] + np.int64(X[r, columns[c]]) - lo[c]] = 0
    for c in range(k):
        rank = 0
        for v in range(table_off[c], table_off[c + 1]):
            if table[v] == 0:
                table[v] = rank
                rank += 1
    codes = np.empty((n, k), dtype=np.int32)
    for r in range(n):
        for c in range(k):
            codes[r, c] = table[table_off[c] + np.int64(X[r, columns[c]]) - lo[c]]
    return codes, table


@nb.njit(cache=True)
def _table_codes(X, columns, lo, table, table_off):
    """Code of every value of the columns ``columns`` in the tables of
    :func:`_integer_codes` (``lo`` as floats), and a mask of the values that
    have none, whose code is 0: those that are not integers, lie outside the
    table of their column or are absent from it."""
    n = X.shape[0]
    k = columns.shape[0]
    codes = np.zeros((n, k), dtype=np.int64)
    unseen = np.zeros((n, k), dtype=np.bool_)
    for r in range(n):
        for c in range(k):
            d = X[r, columns[c]] - lo[c]
            code = -1
            if 0 <= d < table_off[c + 1] - table_off[c]:
                i = np.int64(d)
                if i == d:
                    code = table[table_off[c] + i]
            if code < 0:
                unseen[r, c] = True
            else:
                codes[r, c] = code
    return codes, unseen


@nb.njit(cache=True)
def _categorical_counts(X, y, offsets, cards, size):
    """Count of every (feature, class, value) cell of a categorical Naive
    Bayes: feature ``j`` owns the cells ``offsets[j] + y * cards[j] + v``."""
    flat = np.zeros(size)
    for r in range(X.shape[0]):
        for j in range(X.shape[1]):
            flat[offsets[j] + y[r] * cards[j] + X[r, j]] += 1.0
    return flat
