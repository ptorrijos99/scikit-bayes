"""Building blocks shared by the estimators: the validation and encoding of
the input, the layout of the attributes as children in the compiled tables
and the fit of the discriminative weights."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Number

import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_array, check_X_y, validate_data
from threadpoolctl import threadpool_limits

from ._kernels import (
    DISCRETE,
    GAUSSIAN,
    LOG_ZERO,
    EvaluationInputs,
    _column_profiles,
    _column_ranges,
    _integer_codes,
    _table_codes,
    gaussian_log_norms,
)
from .discretization import _mdlp_cuts

# Number of elements a temporary array of a chunked computation may hold.
_CHUNK_ELEMENTS = 1 << 22

# Undeclared integer-valued attributes with more distinct values than this are
# taken as continuous (ages, counts, amounts) rather than categorical.
_MAX_AUTO_CATEGORIES = 20

# Integers up to this size are floats without rounding
_EXACT_FLOAT = 2.0**53

# Fraction of the largest variance of the Gaussian children added to their
# variances, the default of GaussianNB and MixedNB
_VAR_SMOOTHING = 1e-9


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


def _encode_columns(X, columns, categories):
    """Codes of the columns ``columns`` of ``X`` in their sorted
    ``categories`` and a mask of the values that are not among them (their
    code is 0), each of shape (n_samples, len(columns)): :func:`_encode_values`
    for every column, with the columns whose categories are integers in a
    small range looked up in one compiled pass over the rows."""
    columns = np.asarray(columns, dtype=np.int64)
    codes = np.zeros((X.shape[0], len(columns)), dtype=np.int64)
    unseen = np.zeros(codes.shape, dtype=bool)
    spans = [None] * len(columns)
    if X.dtype.kind in "iuf":
        spans = [_integer_range(cats) for cats in categories]
    # Integers that a float holds exactly, so that the tables can be indexed
    fast = [k for k, s in enumerate(spans) if s and max(-s[0], s[1]) < _EXACT_FLOAT]
    if fast and X.shape[0]:
        lo = np.array([spans[k][0] for k in fast], dtype=np.float64)
        sizes = [spans[k][1] - spans[k][0] + 1 for k in fast]
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        table = np.full(off[-1], -1, dtype=np.int64)
        for j, k in enumerate(fast):
            cats = categories[k]
            table[off[j] + (cats - spans[k][0]).astype(np.int64)] = np.arange(len(cats))
        sub = _table_codes(X, columns[fast], lo, table, off)
        if len(fast) == len(columns):
            return sub
        codes[:, fast], unseen[:, fast] = sub
    for k in set(range(len(columns))).difference(fast):
        codes[:, k], unseen[:, k] = _encode_values(X[:, columns[k]], categories[k])
    return codes, unseen


def _discrete_codes(X, columns):
    """Sorted distinct values of the columns ``columns`` of ``X`` and the code
    of every value among them: a list with the values of each column (as
    :func:`_unique_values`) and the int32 codes, shape (n_samples,
    len(columns)) (as :func:`_encode_values`). The columns that hold integers
    in a small range, in an integer or a floating ``X``, are encoded together
    in one compiled pass over the rows; the rest column by column."""
    columns = np.asarray(columns, dtype=np.int64)
    n = X.shape[0]
    categories = [None] * len(columns)
    fast = np.zeros(len(columns), dtype=bool)
    if n and len(columns) and np.issubdtype(X.dtype, np.signedinteger):
        lo, hi = _column_ranges(X, columns)
        fast = hi - lo <= 4 * n + 1024  # The span rule of _integer_range
    elif n and len(columns) and np.issubdtype(X.dtype, np.floating):
        lo, hi, whole, _ = _column_profiles(X, columns)
        with np.errstate(invalid="ignore"):
            fast = whole & (np.maximum(-lo, hi) < _EXACT_FLOAT)
            fast &= hi - lo <= 4 * n + 1024
        lo = np.where(fast, lo, 0).astype(np.int64)
        hi = np.where(fast, hi, 0).astype(np.int64)
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
    "gaussian" (anything else).

    The values are only counted when the answer depends on their number:
    counting those of a continuous attribute would sort it.
    """
    lo, hi = col.min(), col.max()
    if lo == hi:
        return None
    if np.all((col == lo) | (col == hi)):
        return "bernoulli"
    integer_valued = np.issubdtype(col.dtype, np.integer) or (
        np.issubdtype(col.dtype, np.floating) and np.all(np.mod(col, 1) == 0)
    )
    if not integer_valued or lo < 0:
        return "gaussian"
    if hi - lo < _MAX_AUTO_CATEGORIES:  # No room for more values than that
        return "categorical"
    if len(_unique_values(col)) <= _MAX_AUTO_CATEGORIES:
        return "categorical"
    return "gaussian"


def _is_pandas_frame(X):
    return hasattr(X, "dtypes") and hasattr(X, "iloc")


def _column(X, j):
    return X.iloc[:, j] if _is_pandas_frame(X) else X[:, j]


def _is_category(col):
    return getattr(col.dtype, "name", None) == "category"


def _holds_labels(col, j, name):
    """Whether the column ``col`` (``X[:, j]``) of dtype object, string or
    category holds labels rather than numbers.

    A pandas ``category`` column always does. Otherwise a column holds labels
    when it holds strings; a column of numbers stored as objects is numeric,
    as scikit-learn takes it. Strings mixed with numbers or with other
    objects are rejected, and so are missing values among strings."""
    if _is_category(col):
        return True
    values = col.to_numpy(dtype=object) if hasattr(col, "to_numpy") else col
    types = set(map(type, values))
    if not any(issubclass(t, str) for t in types):
        return False
    if all(issubclass(t, str) for t in types):
        return True
    missing = np.array([_is_missing(v) for v in values], dtype=bool)
    other = {type(v) for v in values[~missing] if not isinstance(v, str)}
    if other:
        kinds = "numbers" if all(issubclass(t, Number) for t in other) else "objects"
        # The words of the error that check_array raises on such objects,
        # which scikit-learn's estimator checks look for
        raise TypeError(
            f"Column {j} of X mixes strings with {kinds}"
            f" ({', '.join(sorted(t.__name__ for t in other))}): every argument"
            " must be a string or a number, and one column holds only one kind."
        )
    _raise_missing(j, name)


def _is_missing(value):
    """Whether ``value`` is None, NaN or a pandas missing value."""
    return (
        value is None
        or (isinstance(value, float) and value != value)
        or type(value).__name__ in ("NAType", "NaTType")
    )


def _raise_missing(j, name):
    raise ValueError(
        f"Input X contains missing values (None or NaN) in its non-numeric column"
        f" {j}.\n{name} does not support missing values yet; impute them first,"
        " for example with sklearn.impute.SimpleImputer(strategy='most_frequent')."
    )


def _label_columns(X, name):
    """Indices of the columns of ``X`` that hold labels (see
    :func:`_holds_labels`), found from the dtypes alone unless a column has
    dtype object. Empty for numeric arrays, sparse matrices and anything that
    is not a pandas DataFrame or a NumPy array."""
    if _is_pandas_frame(X):
        return [
            j
            for j, dtype in enumerate(X.dtypes)
            if dtype.kind in "OSU" and _holds_labels(X.iloc[:, j], j, name)
        ]
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype.kind not in "OSU":
        return []
    if X.dtype.kind in "SU":
        return list(range(X.shape[1]))
    return [j for j in range(X.shape[1]) if _holds_labels(X[:, j], j, name)]


class _LabelEncoder:
    """Integer codes of the columns of ``X`` that hold labels: the position of
    the label among the sorted labels of the column seen in training, -1 for
    any other label.

    A pandas ``category`` column is encoded by its labels too, not by the
    order of its categories, so that the codes do not depend on that order
    and categories absent from the training rows get no code.

    Attributes
    ----------
    columns : list of int
        The columns that hold labels.
    labels : list of ndarray of object
        Sorted training labels of each of them.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit_transform(self, X, name):
        self.labels, codes = [], []
        for j in self.columns:
            col = _column(X, j)
            if _is_category(col):
                cat_codes = col.cat.codes.to_numpy()
                if np.any(cat_codes < 0):
                    _raise_missing(j, name)
                seen = np.unique(cat_codes)
                values = col.cat.categories.to_numpy(dtype=object)[seen]
                order = _label_order(values, j)
                table = np.full(len(col.cat.categories), -1, dtype=np.int64)
                table[seen[order]] = np.arange(len(seen))
                self.labels.append(values[order])
                codes.append(table[cat_codes])
            else:
                values = col.to_numpy(dtype=object) if hasattr(col, "to_numpy") else col
                labels, inverse = np.unique(values, return_inverse=True)
                self.labels.append(labels.astype(object))
                codes.append(inverse.reshape(-1).astype(np.int64))
        self._index = [{v: k for k, v in enumerate(lab)} for lab in self.labels]
        return self._replace(X, codes)

    def transform(self, X, name):
        codes = []
        for j, index in zip(self.columns, self._index):
            col = _column(X, j)
            if _is_category(col):
                cat_codes = col.cat.codes.to_numpy()
                if np.any(cat_codes < 0):
                    _raise_missing(j, name)
                table = np.array(
                    [index.get(v, -1) for v in col.cat.categories], dtype=np.int64
                )
                codes.append(table[cat_codes])
            else:
                values = col.to_numpy(dtype=object) if hasattr(col, "to_numpy") else col
                codes.append(
                    np.fromiter(
                        (index.get(v, -1) for v in values), np.int64, len(values)
                    )
                )
        return self._replace(X, codes)

    def _replace(self, X, codes):
        """``X`` with the columns ``columns`` replaced by ``codes``, in a
        container that ``check_array`` converts as it would the codes given
        by the user."""
        if _is_pandas_frame(X):
            X = X.copy(deep=False)
            for j, c in zip(self.columns, codes):
                X.isetitem(j, c)
            return X
        if X.dtype.kind in "SU":
            return np.column_stack(codes)
        X = X.copy()
        for j, c in zip(self.columns, codes):
            X[:, j] = c
        return X


def _label_order(values, j):
    try:
        return np.argsort(values, kind="stable")
    except TypeError:
        raise TypeError(
            f"The categories of column {j} of X cannot be sorted: every argument"
            " must be a string or a number, and one column holds only one kind."
        ) from None


def _validate_input(estimator, X, y="no_validation", reset=True):
    """:func:`~sklearn.utils.validation.validate_data` for the classifiers,
    which also take columns of labels (see :func:`_validate_labels`). The
    data to predict of a model that discretizes its continuous children come
    back binned, as the model was fitted (see :func:`_discretized_children`).
    """
    out = _validate_labels(estimator, X, y, reset)
    binning = None if reset else getattr(estimator, "_binning", None)
    if binning is None:
        return out
    if isinstance(y, str) and y == "no_validation":
        return binning.transform(out)
    return binning.transform(out[0]), out[1]


def _validate_labels(estimator, X, y, reset):
    """:func:`~sklearn.utils.validation.validate_data` on input that may hold
    columns of labels.

    Columns that hold labels (pandas ``category`` columns and columns of
    strings, see :func:`_label_columns`) are replaced by the codes of a
    :class:`_LabelEncoder`, fitted when ``reset`` and kept in
    ``estimator._label_encoder``; their indices are then those of
    :func:`_label_feature_indices`. Input without such columns, the common
    case, goes to ``validate_data`` as is, so that its validation, errors and
    cost do not change.
    """
    name = type(estimator).__name__
    encoder = None if reset else getattr(estimator, "_label_encoder", None)
    if isinstance(X, (list, tuple)):
        # Converted only to look for labels (NumPy would turn the numbers of
        # a list that also holds strings into strings)
        try:
            array = np.asarray(X)
            if array.dtype.kind in "SU":
                array = np.asarray(X, dtype=object)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.dtype.kind in "OSU":
            X = array
    columns = _label_columns(X, name)
    if encoder is None and not columns:
        if reset:
            estimator._label_encoder = None
        return validate_data(estimator, X, y, reset=reset)
    if encoder is None and not reset:
        raise ValueError(
            f"X has non-numeric values in the columns {columns}, but {name} was"
            " fitted on numeric columns only."
        )

    # The feature names and their number, checked on X as given
    validate_data(estimator, X, y, reset=reset, skip_check_array=True)
    if reset:
        encoder = estimator._label_encoder = _LabelEncoder(columns)
        X = encoder.fit_transform(X, name)
    elif columns != encoder.columns:
        raise ValueError(
            f"The columns of X that hold labels (strings or pandas categories)"
            f" must be those that held them in fit, {encoder.columns}; got"
            f" {columns}. Pass labels, not their codes, to these columns, and"
            " numbers to the others."
        )
    else:
        X = encoder.transform(X, name)
    if isinstance(y, str) and y == "no_validation":
        return check_array(X, input_name="X", estimator=estimator)
    return check_X_y(X, y, estimator=estimator)


def _label_feature_indices(estimator):
    """Columns that held labels when ``estimator`` was fitted, which are
    discrete whatever their values."""
    encoder = getattr(estimator, "_label_encoder", None)
    return [] if encoder is None else list(encoder.columns)


def _check_label_types(label_columns, gaussian_features):
    """Raise when a column of labels is declared Gaussian."""
    bad = sorted(set(label_columns) & set(gaussian_features))
    if bad:
        raise ValueError(
            f"The features {bad} hold labels (strings or pandas categories) and"
            " cannot be in gaussian_features: they are categorical unless"
            " declared Bernoulli."
        )


class _LabelInputMixin:
    """Tags of the classifiers, which take strings through
    :func:`_validate_input`."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        return tags


def _auto_types(X, columns):
    """:func:`_auto_type` of the columns ``columns`` of ``X``, a list, from
    one compiled pass over the rows instead of several over every column."""
    columns = np.asarray(columns, dtype=np.int64)
    if not len(columns) or not X.shape[0] or X.dtype.kind not in "iuf":
        return [_auto_type(X[:, j]) for j in columns]
    lo, hi, whole, distinct = _column_profiles(X, columns)
    types = []
    for k, j in enumerate(columns):
        if max(-lo[k], hi[k]) >= _EXACT_FLOAT:  # Rounded as floats
            types.append(_auto_type(X[:, j]))
        elif distinct[k] == 1:
            types.append(None)
        elif distinct[k] == 2:
            types.append("bernoulli")
        elif not whole[k] or lo[k] < 0:
            types.append("gaussian")
        elif hi[k] - lo[k] < _MAX_AUTO_CATEGORIES:
            types.append("categorical")
        else:
            few = len(_unique_values(X[:, j])) <= _MAX_AUTO_CATEGORIES
            types.append("categorical" if few else "gaussian")
    return types


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


def _fit_bins(col, y_idx, n_classes, n_bins, strategy):
    """Bins of the continuous column ``col`` (2-D, one column): the cut-points
    of Fayyad and Irani when ``strategy`` is ``"mdlp"`` (``n_bins`` is then
    not used), else a fitted :class:`KBinsDiscretizer`. Either has a
    ``transform`` that gives the bin of every value of a column."""
    if strategy == "mdlp":
        return _Cuts(_mdlp_cuts(col.ravel().astype(float), y_idx, n_classes))
    kwargs = {"subsample": None}
    # quantile_method was added in scikit-learn 1.7
    if strategy == "quantile" and parse_version(
        parse_version(sklearn.__version__).base_version
    ) >= parse_version("1.7"):
        kwargs["quantile_method"] = "linear"
    bins = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy=strategy, **kwargs
    )
    with warnings.catch_warnings():
        # Repeated values leave fewer bins than asked for, which the callers
        # account for
        warnings.filterwarnings("ignore", message="Bins whose width are too small")
        return bins.fit(col)


class _Binning:
    """Bins of the continuous attributes of a model that discretizes them as
    children too (``continuous_children="discretized"``): the model is then
    fitted, and evaluated, on a copy of the data in which those columns hold
    the index of the bin of each value, as categorical attributes.

    Attributes
    ----------
    columns : list of int
        The attributes that are binned.
    bins : list
        Their bins, as :func:`_fit_bins` gives them.
    """

    def __init__(self, columns, bins):
        self.columns = columns
        self.bins = bins

    @classmethod
    def fit_transform(cls, X, y_idx, columns, n_bins, strategy):
        """The binning of the columns ``columns`` of ``X`` and the binned
        copy of ``X``."""
        n_classes = int(y_idx.max()) + 1 if len(y_idx) else 0
        bins = [
            _fit_bins(X[:, [j]], y_idx, n_classes, n_bins, strategy) for j in columns
        ]
        binning = cls(list(columns), bins)
        return binning, binning.transform(X)

    def transform(self, X):
        """Copy of ``X`` with the bin of every value in the binned columns.
        Values beyond the range seen in training fall in the first or the
        last bin."""
        X = np.array(X, dtype=np.float64)
        for j, bins in zip(self.columns, self.bins):
            X[:, j] = bins.transform(X[:, [j]]).ravel()
        return X


def _discretized_children(estimator, X, y_idx):
    """Apply ``continuous_children`` to the validated training ``X`` of
    ``estimator``.

    Returns the data to fit on, the columns that are categorical whatever
    their values and the declaration of Gaussian features to use. With
    ``"gaussian"`` these are ``X``, the columns of labels and the
    ``gaussian_features`` of the estimator. With ``"discretized"`` the
    continuous attributes (those declared Gaussian, and the undeclared ones
    typed as such) are binned with ``n_bins`` and ``strategy``, and become
    categorical: none is left Gaussian. The binning is kept in
    ``estimator._binning`` (None when nothing is binned), which
    :func:`_validate_input` applies to the data to predict.
    """
    labels = _label_feature_indices(estimator)
    estimator._binning = None
    # The models without the parameter (GaussianAnJE, GaussianALR) keep
    # their continuous children
    if getattr(estimator, "continuous_children", "gaussian") == "gaussian":
        return X, labels, estimator.gaussian_features
    n_features = X.shape[1]
    gaussian = _feature_indices(
        estimator.gaussian_features, n_features, "gaussian_features"
    )
    _check_label_types(labels, gaussian)
    declared = set(labels) | set(gaussian)
    for name in ("categorical_features", "bernoulli_features"):
        declared |= set(_feature_indices(getattr(estimator, name), n_features, name))
    undeclared = [i for i in range(n_features) if i not in declared]
    types = _auto_types(X, undeclared)
    columns = sorted(
        gaussian + [i for i, t in zip(undeclared, types) if t == "gaussian"]
    )
    if not columns:
        return X, labels, None
    estimator._binning, X = _Binning.fit_transform(
        X, y_idx, columns, estimator.n_bins, estimator.strategy
    )
    return X, labels + columns, None


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
        # The encoders whose number of codes is only known once fitted come
        # first, since that number decides the type of the codes
        self.encoders_ = [None] * X.shape[1]
        n_classes = int(y_idx.max()) + 1 if len(y_idx) else 0
        binned, reused, encoded = [], [], []
        undeclared = [
            i
            for i in range(X.shape[1])
            if i not in declared_gaussian and i not in declared_discrete
        ]
        auto = dict(zip(undeclared, _auto_types(X, undeclared)))
        for i in range(X.shape[1]):
            if i in declared_gaussian:
                continuous = True
            elif i in declared_discrete:
                continuous = False
            else:
                continuous = auto[i] == "gaussian"
            if continuous and self.strategy == "mdlp":
                self.encoders_[i] = _fit_bins(X[:, [i]], y_idx, n_classes, None, "mdlp")
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
            enc = _fit_bins(X[:, [i]], y_idx, n_classes, self.n_bins, self.strategy)
            codes[:, i] = enc.transform(X[:, [i]]).ravel()
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
        discrete = [
            i for i, e in enumerate(self.encoders_) if isinstance(e, np.ndarray)
        ]
        if discrete:
            codes[:, discrete], unseen[:, discrete] = _encode_columns(
                X, discrete, [self.encoders_[i] for i in discrete]
            )
        for i, enc in enumerate(self.encoders_):
            if not isinstance(enc, np.ndarray):
                codes[:, i] = enc.transform(X[:, i].reshape(-1, 1)).ravel()
        np.clip(codes, 0, self.cards_ - 1, out=codes)
        return codes, unseen


class _ChildEncodingMixin:
    """Encoding of the attributes as children, shared by the models that keep
    their conditionals in the tables of :mod:`skbn._kernels` (the AnDE family,
    TAN and KDB). Needs the fitted ``backoff_``, ``_encoder``, ``_pooled_mean``
    and ``_pooled_var``."""

    def _layout(self):
        """Type, offset in the rows of the discrete tables and Gaussian column
        of every attribute, and the number of values of every discrete one."""
        types = self.backoff_.feature_types_
        d = self.n_features_in_
        kind = np.zeros(d, dtype=np.int64)
        cards = np.zeros(d, dtype=np.int64)
        cards[types["categorical"]] = getattr(self.backoff_, "_cat_cardinalities", [])
        cards[types["bernoulli"]] = 2
        discrete = np.flatnonzero(cards)
        kind[discrete] = DISCRETE
        kind[types["gaussian"]] = GAUSSIAN
        voff = np.zeros(d, dtype=np.int64)
        voff[discrete] = np.cumsum(cards[discrete]) - cards[discrete]
        gidx = np.zeros(d, dtype=np.int64)
        gidx[types["gaussian"]] = np.arange(len(types["gaussian"]))
        return kind, voff, gidx, cards

    def _value_codes(self, X, cat_codes=None, dtype=np.int64):
        """Value of every discrete attribute as a child (the category of a
        categorical attribute, the bit of a Bernoulli one), -1 for categories
        never seen in training and 0 for the other attributes. ``cat_codes``
        optionally holds the codes of the categorical attributes of a
        training ``X``, as the back-off model returns them."""
        types = self.backoff_.feature_types_
        codes = np.zeros(X.shape, dtype=dtype)
        if cat_codes is not None:
            codes[:, types["categorical"]] = cat_codes
        elif types["categorical"]:
            cat, unseen = self.backoff_._categorical_codes(X)
            codes[:, types["categorical"]] = np.where(unseen, -1, cat)
        if types["bernoulli"]:
            codes[:, types["bernoulli"]] = (
                X[:, types["bernoulli"]] > self.backoff_.bernoulli_thresholds_
            )
        return codes

    def _backoff_tables(self, kind, voff, cards, n_values):
        """The class-level Naive Bayes in the layout of the component tables:
        its log prior, discrete log-conditionals and Gaussian parameters
        (mean, variance and ``log(2 pi variance)``)."""
        types = self.backoff_.feature_types_
        est = self.backoff_.estimators_
        n_classes = len(self.classes_)
        disc = np.zeros((n_classes, n_values))
        for i, j in enumerate(types["categorical"]):
            disc[:, voff[j] : voff[j] + cards[j]] = est[
                "categorical"
            ].feature_log_prob_[i]
        for i, j in enumerate(types["bernoulli"]):
            log_p = est["bernoulli"].feature_log_prob_[:, i]
            disc[:, voff[j]] = np.log1p(-np.exp(log_p))
            disc[:, voff[j] + 1] = log_p
        disc = np.nan_to_num(disc, nan=LOG_ZERO, neginf=LOG_ZERO)
        if types["gaussian"]:
            mean, var = est["gaussian"].theta_, est["gaussian"].var_
        else:
            mean = var = np.zeros((n_classes, 0))
        logn = gaussian_log_norms(var)
        return self.backoff_.class_log_prior_, disc, mean, var, logn

    def _evaluation_inputs(self, X):
        """Per-sample inputs of the evaluation kernels for a validated ``X``:
        the child value codes, the Gaussian attributes, the uninformed
        log-likelihood of every attribute (the Laplace estimate of an empty
        cell: uniform for a discrete child, the pooled Gaussian of the
        attribute for a continuous one, 0 for an unseen category) and the
        parent codes with their unseen mask."""
        types = self.backoff_.feature_types_
        codes = self._value_codes(X)
        xg = np.ascontiguousarray(X[:, types["gaussian"]], dtype=np.float64)
        uninf = np.zeros(X.shape)
        if types["gaussian"]:
            diff = xg - self._pooled_mean
            uninf[:, types["gaussian"]] = -0.5 * (
                diff**2 / self._pooled_var + np.log(2.0 * np.pi * self._pooled_var)
            )
        kind, _, _, cards = self._layout()
        discrete = np.flatnonzero(kind == DISCRETE)
        uninf[:, discrete] = np.where(
            codes[:, discrete] < 0, 0.0, -np.log(cards[discrete].astype(float))
        )
        parent_codes, unseen = self._encoder.transform(X)
        return EvaluationInputs(codes, xg, uninf, parent_codes, unseen)


# Variables that index a weight at each level. A weight always belongs to one
# term (a component, or a factor of a component); "value" is the value of the
# term's own variable, "class" the class and "parents" the parent
# instantiation of a child factor.
_LEVEL_DIMS = {
    "component": {
        1: (),
        2: ("value",),
        3: ("class",),
        4: ("value", "class"),
    },
    "factor": {
        1: (),
        2: ("value",),
        3: ("value", "class"),
        4: ("value", "class", "parents"),
    },
}


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
