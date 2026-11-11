"""Weighted AnDE: the geometric pool of exponent-weighted components.

The scores and gradients of the discriminative weights of WeightedAnDE,
pooled geometrically or arithmetically, of which WeightedTAN and WeightedKDB
use the geometric pool of their factors, and the pairwise sums that follow
NumPy's order of summation.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numba as nb
import numpy as np


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
