"""Building blocks shared by the AnDE and AnJE families."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

# Number of elements a temporary array of a chunked computation may hold.
_CHUNK_ELEMENTS = 1 << 22

# Undeclared integer-valued attributes with more distinct values than this are
# taken as continuous (ages, counts, amounts) rather than categorical.
_MAX_AUTO_CATEGORIES = 20


def _auto_type(col):
    """Type inferred for an undeclared attribute from its training values:
    None for a constant, then "bernoulli" (two values), "categorical"
    (non-negative integer values, at most ``_MAX_AUTO_CATEGORIES`` of them) or
    "gaussian" (anything else)."""
    n_unique = len(np.unique(col))
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


class _AttributeEncoder:
    """Integer codes of the attributes, as used for parents and joins.

    Continuous attributes (declared Gaussian, or typed Gaussian by
    :func:`_auto_type` when undeclared, the rule of :class:`MixedNB`) are
    binned with :class:`KBinsDiscretizer`; the rest are label-encoded.

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

    def fit_transform(self, X):
        declared_gaussian = {int(i) for i in self.gaussian_features or []}
        declared_discrete = {int(i) for i in self.discrete_features or []}
        kwargs = {"subsample": None}
        # quantile_method was added in scikit-learn 1.7
        if self.strategy == "quantile" and tuple(
            map(int, sklearn.__version__.split(".")[:2])
        ) >= (1, 7):
            kwargs["quantile_method"] = "linear"

        codes = np.zeros(X.shape, dtype=np.int64)
        self.encoders_ = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if i in declared_gaussian:
                continuous = True
            elif i in declared_discrete:
                continuous = False
            else:
                continuous = _auto_type(col) == "gaussian"
            if continuous:
                enc = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode="ordinal",
                    strategy=self.strategy,
                    **kwargs,
                )
                codes[:, i] = enc.fit_transform(col.reshape(-1, 1)).ravel()
            else:
                enc = LabelEncoder()
                codes[:, i] = enc.fit_transform(col)
            self.encoders_.append(enc)
        self.cards_ = codes.max(axis=0) + 1
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
            if isinstance(enc, LabelEncoder):
                known = np.isin(col, enc.classes_)
                unseen[:, i] = ~known
                codes[:, i] = enc.transform(np.where(known, col, enc.classes_[0]))
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
    res = minimize(
        objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options=options
    )
    return res.x, res.nit, res.status != 1


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
