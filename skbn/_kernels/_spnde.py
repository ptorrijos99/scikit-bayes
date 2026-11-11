"""Compiled kernels that count and evaluate the SPnDEs of an ensemble.

Every SPnDE is a Naive Bayes model on the augmented class ``Y* = (y, x_p)``.
Its parameters are stored per augmented state observed in training, in flat
tables shared by all the components of the ensemble, described field by
field in :class:`SPnDETables`; the samples to evaluate come in an
:class:`~skbn._kernels._states.EvaluationInputs`.
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
    _clamp,
    _find,
    _gaussian_ll,
    _Kernel,
)
from ._states import _augmented_keys, _histogram, _state_index

# Augmented states with up to this many training rows keep the rows instead of
# a dense row of estimates
_MAX_LIGHT = 16


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
