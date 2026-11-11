"""Building blocks shared by the AnDE and AnJE families."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.fixes import parse_version
from threadpoolctl import threadpool_limits

from ._kernels import _column_ranges, _integer_codes
from .discretization import _mdlp_cuts

# Number of elements a temporary array of a chunked computation may hold.
_CHUNK_ELEMENTS = 1 << 22

# Undeclared integer-valued attributes with more distinct values than this are
# taken as continuous (ages, counts, amounts) rather than categorical.
_MAX_AUTO_CATEGORIES = 20


def _integer_range(values):
    """``(lo, hi)`` when ``values`` are integers spanning at most
    ``4 * len(values) + 1024`` consecutive values (so that they can index a
    table), else None."""
    if values.size == 0:
        return None
    if np.issubdtype(values.dtype, np.integer):
        lo, hi = values.min(), values.max()
    elif np.issubdtype(values.dtype, np.floating):
        lo, hi = values.min(), values.max()
        if not (np.isfinite(lo) and np.isfinite(hi)) or np.any(values % 1):
            return None
    else:
        return None
    lo, hi = int(lo), int(hi)
    if hi - lo > 4 * values.size + 1024:
        return None
    return lo, hi


def _unique_values(col):
    """``np.unique(col)``, counted with a table when ``col`` holds integers in
    a small range."""
    span = _integer_range(col)
    if span is None:
        return np.unique(col)
    lo, hi = span
    present = np.bincount((col - lo).astype(np.int64), minlength=hi - lo + 1)
    return (np.flatnonzero(present) + lo).astype(col.dtype)


def _encode_values(col, categories):
    """Codes of ``col`` in the sorted array ``categories`` and a mask of the
    values that are not in it (their code is 0)."""
    span = _integer_range(categories)
    if span is None or not (
        np.issubdtype(col.dtype, np.integer) or np.issubdtype(col.dtype, np.floating)
    ):
        pos = np.minimum(np.searchsorted(categories, col), len(categories) - 1)
        known = categories[pos] == col
        return np.where(known, pos, 0), ~known
    lo, hi = span
    table = np.full(hi - lo + 2, -1, dtype=np.int64)
    table[(categories - lo).astype(np.int64)] = np.arange(len(categories))
    # Values outside [lo, hi] or not integers go to the last, empty, entry
    offset = col - lo
    inside = (offset >= 0) & (offset <= hi - lo)
    if np.issubdtype(col.dtype, np.floating):
        inside[inside] = offset[inside] % 1 == 0
    codes = table[np.where(inside, offset, hi - lo + 1).astype(np.int64)]
    known = codes >= 0
    return np.where(known, codes, 0), ~known


def _discrete_codes(X, columns):
    """Sorted distinct values of the columns ``columns`` of ``X`` and the code
    of every value among them: a list with the values of each column (as
    :func:`_unique_values`) and the int32 codes, shape (n_samples,
    len(columns)) (as :func:`_encode_values`). Integer columns whose values span a small
    range are encoded together in one compiled pass over the rows; the rest
    column by column."""
    columns = np.asarray(columns, dtype=np.int64)
    n = X.shape[0]
    categories = [None] * len(columns)
    fast = np.zeros(len(columns), dtype=bool)
    if n and len(columns) and np.issubdtype(X.dtype, np.signedinteger):
        lo, hi = _column_ranges(X, columns)
        fast = hi - lo <= 4 * n + 1024  # The span rule of _integer_range
    idx = np.flatnonzero(fast)
    codes = np.empty((n, len(columns)), dtype=np.int32)
    if len(idx):
        span = hi[idx] - lo[idx] + 1
        off = np.concatenate([[0], np.cumsum(span)]).astype(np.int64)
        sub, table = _integer_codes(X, columns[idx], lo[idx], off)
        if len(idx) == len(columns):
            codes = sub
        else:
            codes[:, idx] = sub
        for k, c in enumerate(idx):
            present = np.flatnonzero(table[off[k] : off[k + 1]] >= 0)
            categories[c] = (present + lo[c]).astype(X.dtype)
    for c in np.flatnonzero(~fast):
        col = X[:, columns[c]]
        categories[c] = _unique_values(col)
        codes[:, c] = _encode_values(col, categories[c])[0]
    return categories, codes


def _auto_type(col):
    """Type inferred for an undeclared attribute from its training values:
    None for a constant, then "bernoulli" (two values), "categorical"
    (non-negative integer values, at most ``_MAX_AUTO_CATEGORIES`` of them) or
    "gaussian" (anything else)."""
    n_unique = len(_unique_values(col))
    if n_unique == 1:
        return None
    if n_unique == 2:
        return "bernoulli"
    integer_valued = np.issubdtype(col.dtype, np.integer) or (
        np.issubdtype(col.dtype, np.floating) and np.all(np.mod(col, 1) == 0)
    )
    if integer_valued and np.all(col >= 0) and n_unique <= _MAX_AUTO_CATEGORIES:
        return "categorical"
    return "gaussian"


def _row_chunks(n_rows, row_size):
    """Slices over ``n_rows`` rows such that each chunk holds about
    ``_CHUNK_ELEMENTS`` elements when a row has ``row_size`` of them."""
    step = max(1, _CHUNK_ELEMENTS // max(int(row_size), 1))
    for start in range(0, n_rows, step):
        yield slice(start, min(start + step, n_rows))


def _feature_indices(indices, n_features, name):
    """Feature indices given as integers or as a boolean mask, as a list."""
    if indices is None:
        return []
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError(f"'{name}' must be a 1D array-like.")
    if indices.dtype == bool:
        if len(indices) != n_features:
            raise ValueError(
                f"The boolean mask '{name}' must have length {n_features}, got"
                f" {len(indices)}."
            )
        return np.flatnonzero(indices).tolist()
    if indices.size and not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"'{name}' must hold integer indices or be a boolean mask.")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= n_features):
        raise ValueError(f"All indices in '{name}' must be in [0, {n_features - 1}].")
    return indices.tolist()


class _Cuts:
    """Cut-points of a continuous attribute, whose codes are the intervals
    they delimit."""

    def __init__(self, cuts):
        self.cuts = cuts

    def transform(self, col):
        return np.digitize(col.ravel(), self.cuts)


class _AttributeEncoder:
    """Integer codes of the attributes, as used for parents and joins.

    Continuous attributes (declared Gaussian, or typed Gaussian by
    :func:`_auto_type` when undeclared, the rule of :class:`MixedNB`) are
    binned: with :class:`KBinsDiscretizer`, or at the cut-points of Fayyad
    and Irani when ``strategy`` is ``"mdlp"`` (``n_bins`` is then not used).
    The rest are label-encoded.

    Attributes
    ----------
    cards_ : ndarray of shape (n_features,)
        Number of codes of each attribute seen in training.
    """

    def __init__(
        self, n_bins, strategy, gaussian_features=None, discrete_features=None
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.gaussian_features = gaussian_features
        self.discrete_features = discrete_features

    def fit_transform(self, X, y_idx, known=None, dtype=np.int64):
        """Fit the encoders and return the codes of ``X``, of type ``dtype``
        (with None, 16 bits when every attribute fits in them, else 32).
        ``y_idx`` holds the class indices, which only the ``"mdlp"`` strategy
        looks at. ``known`` optionally gives ``(columns, categories, codes)``,
        the output of :func:`_discrete_codes` for some discrete columns,
        reused as is; the other discrete columns are encoded together."""
        known_columns = {}
        if known is not None:
            known_columns = {int(c): k for k, c in enumerate(known[0])}
        declared_gaussian = {int(i) for i in self.gaussian_features or []}
        declared_discrete = {int(i) for i in self.discrete_features or []}
        kwargs = {"subsample": None}
        # quantile_method was added in scikit-learn 1.7
        sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)
        if self.strategy == "quantile" and sklearn_version >= parse_version("1.7"):
            kwargs["quantile_method"] = "linear"

        # The encoders whose number of codes is only known once fitted come
        # first, since that number decides the type of the codes
        self.encoders_ = [None] * X.shape[1]
        n_classes = int(y_idx.max()) + 1 if len(y_idx) else 0
        binned, reused, encoded = [], [], []
        for i in range(X.shape[1]):
            if i in declared_gaussian:
                continuous = True
            elif i in declared_discrete:
                continuous = False
            else:
                continuous = _auto_type(X[:, i]) == "gaussian"
            if continuous and self.strategy == "mdlp":
                col = X[:, i].astype(float)
                self.encoders_[i] = _Cuts(_mdlp_cuts(col, y_idx, n_classes))
            elif continuous:
                binned.append(i)
            elif i in known_columns:
                self.encoders_[i] = known[1][known_columns[i]]
                reused.append(i)
            else:
                # The sorted values of a discrete attribute
                encoded.append(i)
        if encoded:
            categories, encoded_codes = _discrete_codes(X, encoded)
            for i, values in zip(encoded, categories):
                self.encoders_[i] = values
        if dtype is None:
            widest = max(
                [self.n_bins, 2]
                + [
                    len(enc) if isinstance(enc, np.ndarray) else len(enc.cuts) + 1
                    for enc in self.encoders_
                    if enc is not None
                ]
            )
            dtype = np.uint16 if widest <= 1 << 16 else np.int32

        codes = np.zeros(X.shape, dtype=dtype)
        for i, enc in enumerate(self.encoders_):
            if isinstance(enc, _Cuts):
                codes[:, i] = enc.transform(X[:, i])
        for i in binned:
            enc = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy=self.strategy, **kwargs
            )
            with warnings.catch_warnings():
                # Repeated values leave fewer bins than asked for, which
                # cards_ accounts for
                warnings.filterwarnings(
                    "ignore", message="Bins whose width are too small"
                )
                codes[:, i] = enc.fit_transform(X[:, [i]]).ravel()
            self.encoders_[i] = enc
        if reused:
            codes[:, reused] = known[2][:, [known_columns[i] for i in reused]]
        if encoded:
            codes[:, encoded] = encoded_codes
        self.cards_ = codes.max(axis=0).astype(np.int64) + 1
        return codes

    def transform(self, X):
        """Codes of ``X`` and a mask of discrete values never seen in training.

        Codes are clipped to the training range; the code of an unseen value
        is meaningless and callers decide how to treat it through the mask.
        """
        codes = np.zeros(X.shape, dtype=np.int64)
        unseen = np.zeros(X.shape, dtype=bool)
        for i, enc in enumerate(self.encoders_):
            col = X[:, i]
            if isinstance(enc, np.ndarray):
                codes[:, i], unseen[:, i] = _encode_values(col, enc)
            else:
                codes[:, i] = enc.transform(col.reshape(-1, 1)).ravel()
        np.clip(codes, 0, self.cards_ - 1, out=codes)
        return codes, unseen


def _fit_weights(
    nll_and_grad,
    theta0,
    l2_reg,
    max_iter,
    log_space=False,
    bounds=None,
    tol=None,
    gtol=None,
    maxcor=10,
):
    """Minimise a conditional negative log-likelihood over discriminative weights.

    The objective is ``NLL(theta) + l2_reg * ||w - 1||^2`` with ``w = theta``,
    or ``w = exp(theta)`` when ``log_space``; the penalty pulls the weights
    towards the generative model (``w = 1``). Solved with L-BFGS-B using the
    analytic gradient returned by ``nll_and_grad``. ``tol`` sets both the
    relative-reduction and the projected-gradient stopping tolerances (SciPy's
    defaults when None), ``gtol`` overrides the second one and ``maxcor`` is
    the number of corrections kept by L-BFGS-B.

    Returns
    -------
    theta : ndarray
        Optimal parameters.
    n_iter : int
        Number of L-BFGS-B iterations.
    converged : bool
        False when the optimiser stopped at its iteration or evaluation limit.
    """

    def objective(theta):
        nll, grad = nll_and_grad(theta)
        w = np.exp(theta) if log_space else theta
        dreg = 2.0 * l2_reg * (w - 1.0)
        if log_space:
            dreg *= w
        return nll + l2_reg * np.sum((w - 1.0) ** 2), grad + dreg

    options = {"maxiter": max_iter, "maxcor": maxcor}
    if tol is not None:
        options.update(ftol=tol, gtol=tol)
    if gtol is not None:
        options["gtol"] = gtol
    # L-BFGS-B works on vectors of the size of the weights, too small for
    # threads: those of BLAS only slow it down (many times on a busy CPU) and
    # make the sums depend on their number
    with threadpool_limits(limits=1, user_api="blas"):
        res = minimize(
            objective,
            theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
    return res.x, res.nit, res.status != 1


def _pad_last(theta, n_extra):
    """``theta`` followed by ``n_extra`` copies of its last entry, the weight
    of the keys unseen in training. The weighted kernels read the weight of
    class ``c`` of a term ``c`` places after that of class 0, which for that
    last weight must be itself."""
    if not n_extra:
        return theta
    return np.concatenate([theta, np.full(n_extra, theta[-1])])


def _fold_padding(grad, size):
    """Gradient with respect to the ``size`` weights from that with respect
    to their padding by :func:`_pad_last`."""
    if len(grad) > size:
        grad[size - 1] += grad[size:].sum()
        grad = grad[:size]
    return grad


def _warn_not_converged(n_failed, n_problems, max_iter):
    """One :class:`~sklearn.exceptions.ConvergenceWarning` for the weight
    problems of a fit that stopped at ``max_iter``."""
    if not n_failed:
        return
    where = "" if n_problems == 1 else f" in {n_failed} of {n_problems} components"
    warnings.warn(
        (
            f"L-BFGS-B reached max_iter={max_iter} before converging{where};"
            " increase max_iter or tol."
        ),
        ConvergenceWarning,
    )
