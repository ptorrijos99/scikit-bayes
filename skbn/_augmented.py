"""
Augmented Naive Bayes networks: the base of TAN and KDB.

In these networks every attribute depends on the class and on a few other
attributes, its parents, chosen by a structure learner (the spanning tree of
:mod:`skbn.tan`, the k-dependence rule of :mod:`skbn.kdb`). This module holds
what both learners share: the conditional mutual information between
attributes given the class and the snapping of near ties on which their
structures are chosen, the generative fit and evaluation of the network
(``_BaseAugmentedNB``), and its discriminative exponent weights
(``_WeightedAugmentedNB``, the base of WeightedTAN and WeightedKDB).
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import Bunch
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from ._core import (
    _LEVEL_DIMS,
    _VAR_SMOOTHING,
    _AttributeEncoder,
    _ChildEncodingMixin,
    _discretized_children,
    _fit_weights,
    _fold_padding,
    _LabelInputMixin,
    _pad_last,
    _row_chunks,
    _validate_input,
    _warn_not_converged,
)
from ._kernels import (
    DISCRETE,
    GAUSSIAN,
    IGNORED,
    LOG_ZERO,
    ConditionalTables,
    _threads,
    _weighted_geometric_gradient,
    _weighted_geometric_scores,
    conditional_mutual_info,
    conditional_scores,
    count_conditionals,
    count_states,
    gaussian_log_norms,
)
from .mixed_nb import MixedNB


def _conditional_mutual_info(codes, y_idx, cards, n_classes):
    """Empirical conditional mutual information I(A_i; A_j | C) (in nats).

    Parameters
    ----------
    codes : ndarray of shape (n_samples, n_features)
        Integer codes of the (discretized) attributes, in ``[0, cards[j])``.
    y_idx : ndarray of shape (n_samples,)
        Class indices in ``[0, n_classes)``.
    cards : ndarray of shape (n_features,)
        Number of values of each attribute.
    n_classes : int
        Number of classes.

    Returns
    -------
    cmi : ndarray of shape (n_features, n_features)
        Symmetric matrix with zeros on the diagonal.
    """
    return conditional_mutual_info(
        np.ascontiguousarray(codes.T, dtype=np.int32),
        np.asarray(y_idx, dtype=np.int64),
        np.asarray(cards, dtype=np.int64),
        n_classes,
    )


# Weights closer than this fraction of the largest one are taken as equal when
# a structure is chosen. Attributes that are copies or relabellings of each
# other have the same mutual information in exact arithmetic but come out a
# few ulps apart, and a comparison between them would follow the rounding of
# the platform instead of the documented rule for ties.
_TIE_RTOL = 1e-10


def _snap_ties(values, tol):
    """``values`` (1-D) with every run of values closer than ``tol`` to their
    neighbour replaced by the smallest of the run, so that exact comparisons
    see them as ties."""
    if len(values) < 2:
        return values
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    first = np.concatenate([[0], np.flatnonzero(np.diff(ranked) > tol) + 1])
    run = np.searchsorted(first, np.arange(len(ranked)), side="right") - 1
    snapped = np.empty_like(ranked)
    snapped[order] = ranked[first[run]]
    return snapped


class _BaseAugmentedNB(
    _LabelInputMixin, _ChildEncodingMixin, ClassifierMixin, BaseEstimator
):
    """Naive Bayes whose attributes may also have other attributes as parents
    (TAN, KDB).

    Subclasses learn the attribute parents in ``_learn_structure``. Every
    conditional :math:`P(x_i \\mid y, \\mathbf{x}_{\\Pi(i)})` is counted on the
    augmented class :math:`(y, \\mathbf{x}_{\\Pi(i)})` and stored in the flat
    tables of :mod:`skbn._kernels`; the attributes without attribute parents
    take their conditionals from the class-level Naive Bayes ``backoff_``.
    """

    def _fit(self, X, y):
        """Learn the structure and count the conditionals; returns the
        validated ``X`` and the class indices of ``y``."""
        X, y = _validate_input(self, X, y)
        self.classes_ = unique_labels(y)
        self._check_structure_params()
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)

        # --- 1. Class-level model: prior, parentless conditionals, types ---
        X, discrete, gaussian = _discretized_children(self, X, y_idx)
        self.backoff_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            bernoulli_features=self.bernoulli_features,
            gaussian_features=gaussian,
            var_prior_weight=self.var_prior_weight,
        )._fit_encoded(X, y_idx, self.classes_, discrete)
        types = self.backoff_.feature_types_

        # --- 2. Discretize attributes (structure learning and parents), with
        #        the types the children use ---
        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=types["gaussian"],
            discrete_features=types["categorical"] + types["bernoulli"],
        )
        codes = self._encoder.fit_transform(X, y_idx)
        gauss = types["gaussian"]
        Xg = np.asarray(X[:, gauss], dtype=float) if gauss else np.zeros((len(X), 0))
        self._pooled_mean = Xg.mean(axis=0) if gauss else np.zeros(0)
        self._pooled_var = Xg.var(axis=0) + (
            self.backoff_.estimators_["gaussian"].epsilon_ if gauss else 0.0
        )

        # --- 3. Structure, then the conditionals of every attribute ---
        with _threads(self.n_jobs):
            parents = self._learn_structure(
                codes, y_idx, self._encoder.cards_, n_classes
            )
            self._fit_conditionals(X, y_idx, codes, parents)
        return X, y_idx

    def _fit_conditionals(self, X, y_idx, codes, parents):
        """Count ``P(x_i | y, x_parents[i])`` for every attribute that is not
        ignored, and copy the parentless ones from ``backoff_``."""
        types = self.backoff_.feature_types_
        n_classes = len(self.classes_)
        kind, _, gidx, cards = self._layout()
        pcards = self._encoder.cards_
        children = np.flatnonzero(kind != IGNORED).astype(np.int64)
        n_factors = len(children)
        n_parents = np.array([len(parents[i]) for i in children], dtype=np.int64)
        width = max(1, int(n_parents.max(initial=0)))
        pa = np.zeros((n_factors, width), dtype=np.int64)
        strides = np.zeros((n_factors, width), dtype=np.int64)
        n_combos = np.ones(n_factors, dtype=np.int64)
        for m, i in enumerate(children):
            # Mixed radix with the first parent most significant
            for q in range(n_parents[m] - 1, -1, -1):
                pa[m, q] = parents[i][q]
                strides[m, q] = n_combos[m]
                if n_classes * float(n_combos[m]) * pcards[pa[m, q]] >= 2**62:
                    raise ValueError(
                        f"Attribute {i} has {n_parents[m]} parents whose joint"
                        " states with the class exceed a 64-bit index; use fewer"
                        " parents or fewer bins."
                    )
                n_combos[m] *= pcards[pa[m, q]]

        xg = np.ascontiguousarray(X[:, types["gaussian"]], dtype=np.float64)
        value_codes = self._value_codes(X)
        codes_t = np.ascontiguousarray(codes.T)
        child_cards = np.where(kind[children] == DISCRETE, cards[children], 0)
        # A discrete child whose table (states x values) would be much larger
        # than the training data keeps only its non-empty cells (n_cells > 0)
        n_states, n_cells = count_states(
            y_idx,
            codes_t,
            pa,
            strides,
            n_combos,
            n_classes,
            value_codes,
            children,
            child_cards,
        )
        state_off = np.concatenate([[0], np.cumsum(n_states)]).astype(np.int64)
        cell_off = np.concatenate([[0], np.cumsum(n_cells)]).astype(np.int64)
        disc_size = np.where(n_cells > 0, n_cells, n_states * child_cards)
        disc_off = np.concatenate([[0], np.cumsum(disc_size)]).astype(np.int64)
        keys, count, disc, cell_keys, mean, var = count_conditionals(
            y_idx,
            codes_t,
            value_codes,
            xg,
            pa,
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
        )

        # Discrete children: log((N(s, v) + alpha) / (N(s) + alpha * card)),
        # a factor at a time; unseen[s] is the value of N(s, v) = 0 for the
        # factors that keep only their non-empty cells
        unseen = np.zeros(len(count) if n_cells.any() else 0)
        with np.errstate(divide="ignore"):
            for m in np.flatnonzero(child_cards):
                o, card = state_off[m], child_cards[m]
                lo, hi = disc_off[m], disc_off[m + 1]
                if n_cells[m]:
                    local = cell_keys[cell_off[m] : cell_off[m + 1]] // card
                    states = slice(o, state_off[m + 1])
                    unseen[states] = np.log(np.zeros(n_states[m]) + self.alpha) - (
                        np.log(count[states] + self.alpha * card)
                    )
                else:
                    local = np.arange(hi - lo) // card
                disc[lo:hi] = np.log(disc[lo:hi] + self.alpha) - np.log(
                    count[o + local] + self.alpha * card
                )
        np.nan_to_num(disc, copy=False, nan=LOG_ZERO, neginf=LOG_ZERO)
        np.nan_to_num(unseen, copy=False, nan=LOG_ZERO, neginf=LOG_ZERO)

        # Gaussian children: the variance of GaussianNB on the attribute (plus
        # var_smoothing times its variance), shrunk towards that variance
        if xg.shape[1]:
            factor = np.repeat(np.arange(n_factors), n_states)
            is_gauss = kind[children[factor]] == GAUSSIAN
            col_var = np.var(xg, axis=0)[gidx[children[factor]]]
            epsilon = _VAR_SMOOTHING * col_var
            nu = self.var_prior_weight
            shrunk = (count * (var + epsilon) + nu * (col_var + epsilon)) / (count + nu)
            var = np.where(is_gauss, shrunk, var)

        # A Gaussian attribute without attribute parents takes the variance of
        # backoff_, smoothed with the largest variance of all the Gaussian
        # attributes (the discrete ones already match it)
        for m in np.flatnonzero(n_parents == 0):
            j = children[m]
            if kind[j] == GAUSSIAN:
                o = state_off[m]
                var[o : o + n_classes] = self.backoff_.estimators_["gaussian"].var_[
                    :, gidx[j]
                ]
        # log(2 pi var) of the Gaussian children, once per fit rather than at
        # every evaluation
        logn = np.zeros(len(var))
        if xg.shape[1]:
            gauss = np.repeat(kind[children] == GAUSSIAN, n_states)
            logn[gauss] = gaussian_log_norms(var[gauss, None])[:, 0]

        self._state_count = count
        self._tables = ConditionalTables(
            parents=pa,
            strides=strides,
            n_parents=n_parents,
            children=children,
            n_combos=n_combos,
            state_off=state_off,
            state_keys=keys,
            disc_off=disc_off,
            disc=disc,
            cell_off=cell_off,
            cell_keys=cell_keys,
            cell_unseen=unseen,
            mean=mean,
            var=var,
            logn=logn,
            kind=kind,
            gidx=gidx,
            cards=cards,
        )

    @property
    def feature_types_(self):
        """Indices of the attributes of each type the model uses, under the
        keys "gaussian", "categorical" and "bernoulli"."""
        check_is_fitted(self, attributes=["backoff_"])
        return self.backoff_.feature_types_

    @property
    def edges_(self):
        """Arcs between attributes, as ``(parent, child)`` pairs of feature
        indices in the order of the children. Every attribute also has the
        class as a parent, which the pairs leave out."""
        check_is_fitted(self, attributes=["parents_"])
        return [
            (int(p), i)
            for i, pa in enumerate(self.parents_)
            for p in np.atleast_1d(pa)
            if p >= 0
        ]

    def to_dot(self, feature_names=None, class_name="class"):
        """
        Return the network in the DOT language of Graphviz.

        The text can be rendered with Graphviz or read by the libraries that
        take DOT, such as ``networkx`` and ``pydot``. The attributes the
        model ignores (constant in training) are left out.

        Parameters
        ----------
        feature_names : array-like of str of shape (n_features,), default=None
            Names of the attributes. By default ``feature_names_in_``, or
            ``x0``, ``x1``, ... when the model was fitted without names.

        class_name : str, default="class"
            Name of the class node.

        Returns
        -------
        dot : str
            A ``digraph`` with an arc from the class to every attribute and
            the arcs of ``edges_``.
        """
        check_is_fitted(self, attributes=["parents_"])
        if feature_names is None:
            feature_names = getattr(
                self,
                "feature_names_in_",
                [f"x{i}" for i in range(self.n_features_in_)],
            )
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"feature_names must have {self.n_features_in_} names, got"
                f" {len(feature_names)}."
            )

        def quoted(name):
            return '"' + str(name).replace("\\", "\\\\").replace('"', '\\"') + '"'

        used = set(self._tables.children.tolist())
        lines = [
            f"    {quoted(class_name)} -> {quoted(feature_names[i])};"
            for i in sorted(used)
        ]
        lines += [
            f"    {quoted(feature_names[p])} -> {quoted(feature_names[i])};"
            for p, i in self.edges_
            if p in used and i in used
        ]
        return "\n".join(["digraph {", *lines, "}"]) + "\n"

    def _factor(self, feature):
        """Index of an attribute, given by its index or by its name, and the
        position of its conditional among the factors of the network."""
        check_is_fitted(self, attributes=["parents_"])
        if isinstance(feature, str):
            names = getattr(self, "feature_names_in_", None)
            if names is None or feature not in names:
                raise ValueError(
                    f"{feature!r} is not the name of a feature seen in fit."
                )
            feature = int(np.flatnonzero(names == feature)[0])
        elif not (isinstance(feature, Integral) and 0 <= feature < self.n_features_in_):
            raise ValueError(
                "feature must be the name of a feature seen in fit or its index in"
                f" [0, {self.n_features_in_ - 1}], got {feature!r}."
            )
        where = np.flatnonzero(self._tables.children == feature)
        if not len(where):
            raise ValueError(
                f"Feature {feature} is constant in training: the model ignores it"
                " and it has no conditional."
            )
        return int(feature), int(where[0])

    def _parent_states(self, m, parent_keys):
        """Parents of factor ``m`` and their values, of shape (len(keys),
        n_parents), in the parent states of keys ``parent_keys``."""
        t = self._tables
        parents = [int(p) for p in t.parents[m, : t.n_parents[m]]]
        values = np.empty((len(parent_keys), len(parents)), dtype=object)
        for q, p in enumerate(parents):
            codes = parent_keys // t.strides[m, q] % self._encoder.cards_[p]
            values[:, q] = self._parent_values(p)[codes]
        return parents, values

    def _feature_name(self, j):
        names = getattr(self, "feature_names_in_", None)
        return f"x{j}" if names is None else names[j]

    @staticmethod
    def _table_frame(data, index_values, index_names, columns=None):
        """A DataFrame of ``data`` whose rows are indexed by the arrays
        ``index_values``, named ``index_names``; a single row when there is
        nothing to tell the rows apart."""
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("as_frame=True requires pandas.") from error
        if index_values:
            index = pd.MultiIndex.from_arrays(index_values, names=index_names)
        else:
            index = pd.Index(["all"])
        return pd.DataFrame(data, index=index, columns=columns)

    def conditional_table(self, feature, as_frame=False):
        """
        Return the conditional distribution of an attribute in the network.

        The table has a row for every state of the class and of the parents
        of the attribute that occurs in training, with the probability of
        each value of the attribute, or the mean and the variance of its
        Gaussian. The states that do not occur have no row: there the model
        uses the uninformed estimate described in the class docstring.

        Parameters
        ----------
        feature : int or str
            The attribute, by its index or by its name in
            ``feature_names_in_``.

        as_frame : bool, default=False
            If True, return a pandas DataFrame indexed by the class and the
            values of the parents, with a column per value of the attribute
            (or the columns ``mean`` and ``var``).

        Returns
        -------
        table : :class:`~sklearn.utils.Bunch` or DataFrame
            With ``as_frame=False``, a dictionary-like object with:

            feature : int
                Index of the attribute.
            parents : list of int
                Its parents among the attributes.
            classes : ndarray of shape (n_states,)
                Class of every row.
            parent_values : ndarray of shape (n_states, n_parents)
                Values of the parents in every row: their categories, or the
                index of the bin of a continuous parent (see
                ``cut_points_``).
            count : ndarray of shape (n_states,)
                Number of training samples in the state of every row.
            feature_values : ndarray of shape (n_values,) or None
                Values of a discrete attribute: its categories, or False and
                True for a Bernoulli attribute (above its threshold or not).
            proba : ndarray of shape (n_states, n_values) or None
                Probability of every value in every row.
            mean, var : ndarray of shape (n_states,) or None
                Mean and variance of a Gaussian attribute in every row.

        Examples
        --------
        >>> import numpy as np
        >>> from skbn import KDB
        >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5)
        >>> y = np.array([0, 1, 1, 0] * 5)
        >>> table = KDB(k=1).fit(X, y).conditional_table(1)
        >>> table.parents, table.classes.tolist(), table.parent_values.ravel().tolist()
        ([0], [0, 0, 1, 1], [0, 1, 0, 1])
        >>> table.proba.round(2)
        array([[0.86, 0.14],
               [0.14, 0.86],
               [0.14, 0.86],
               [0.86, 0.14]])
        """
        feature, m = self._factor(feature)
        conditional = self._conditional(feature)
        class_index, parent_keys = np.divmod(
            conditional["keys"], self._tables.n_combos[m]
        )
        parents, parent_values = self._parent_states(m, parent_keys)
        table = Bunch(
            feature=feature,
            parents=parents,
            classes=self.classes_[class_index],
            parent_values=parent_values,
            count=conditional["count"],
            feature_values=self._child_values(feature),
            proba=None,
            mean=conditional.get("mean"),
            var=conditional.get("var"),
        )
        if "log_prob" in conditional:
            table.proba = np.exp(conditional["log_prob"])
        if not as_frame:
            return table
        index = [table.classes] + list(parent_values.T)
        names = ["class"] + [self._feature_name(p) for p in parents]
        if table.proba is None:
            return self._table_frame(
                {"mean": table.mean, "var": table.var}, index, names
            )
        frame = self._table_frame(table.proba, index, names, table.feature_values)
        return frame.rename_axis(columns=self._feature_name(feature))

    def _conditional(self, i):
        """The conditional of attribute ``i``: its augmented-state keys
        ``class_index * n_parent_combos + parent_key``, their training counts,
        and the log-probabilities of shape (n_states, n_values) of a discrete
        child or the mean and variance of a Gaussian one."""
        t = self._tables
        m = int(np.flatnonzero(t.children == i)[0])
        lo, hi = t.state_off[m], t.state_off[m + 1]
        out = {"keys": t.state_keys[lo:hi], "count": self._state_count[lo:hi]}
        values = t.disc[t.disc_off[m] : t.disc_off[m + 1]]
        card = t.cards[i]
        if t.kind[i] == DISCRETE and t.cell_off[m + 1] > t.cell_off[m]:
            # Only the non-empty cells are stored
            log_prob = np.repeat(t.cell_unseen[lo:hi], card)
            log_prob[t.cell_keys[t.cell_off[m] : t.cell_off[m + 1]]] = values
            out["log_prob"] = log_prob.reshape(-1, card)
        elif t.kind[i] == DISCRETE:
            out["log_prob"] = values.reshape(-1, card)
        else:
            out["mean"], out["var"] = t.mean[lo:hi], t.var[lo:hi]
        return out

    def _joint_log_likelihood(self, X):
        """Computes log P(y, x) for every sample and class.

        Returns
        -------
        jll : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, attributes=["classes_", "parents_"])
        X = _validate_input(self, X, reset=False)
        with _threads(self.n_jobs):
            jll = conditional_scores(
                self._evaluation_inputs(X), self._tables, len(self.classes_), False
            )
        return jll[:, :, 0] + self.backoff_.class_log_prior_

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class.
        """
        jll = self._joint_log_likelihood(X)
        log_prob = jll - logsumexp(jll, axis=1, keepdims=True)
        return np.nan_to_num(log_prob, nan=-np.log(jll.shape[1]))

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self, attributes=["classes_", "parents_"])
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class _WeightedAugmentedNB:
    """Discriminative exponent weights on the terms of a network of
    :class:`_BaseAugmentedNB`, the weighted CCBN of Zaidi et al. (2017):
    ``log P^w(y, x) = sum_t w_t log theta_t`` over the class prior and the
    conditional of every factor, with the variables of ``weight_level``
    (``_LEVEL_DIMS["factor"]``) indexing each weight.

    The network is a single component of the geometric factor-weighted
    ensemble of :class:`WeightedAnDE`, so the objective is evaluated by the
    same kernels: the log-terms, shape (n, n_classes, n_terms), are the values
    of its operator and the weight of every term its indices. As there, only
    the keys seen in training get a weight (a key is seen with every class
    once its value and parents occur), and one more weight, never trained,
    stands for every other key and keeps its generative value of 1.
    """

    _weight_constraints: dict = {
        "weight_level": [Interval(Integral, 1, 4, closed="both")],
        "l2_reg": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left"), None],
    }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the network, then learn the discriminative weights.

        Learns the structure and counts the conditional probability tables as
        the unweighted model does, then optimizes the weights with L-BFGS-B on
        the conditional log-likelihood, starting at the generative model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y_idx = self._fit(X, y)
        inputs = self._evaluation_inputs(X)
        self._setup_weights(self._term_keys(inputs))
        terms, index = self._design(inputs)
        del inputs
        Y = np.eye(len(self.classes_))[y_idx]
        weights, self.n_iter_, converged = _fit_weights(
            lambda w: self._nll_and_grad(w, terms, index, Y),
            np.ones(self.n_weights_),
            self.l2_reg,
            self.max_iter,
            tol=self.tol,
        )
        _warn_not_converged(not converged, 1, self.max_iter)
        self.learned_weights_ = weights
        return self

    def _term_weights(self, t):
        """Weights of term ``t`` (0 is the class prior, then the factors), a
        row per key seen in training and a column per class when the level
        has it."""
        check_is_fitted(self, attributes=["learned_weights_"])
        lo, hi = self._key_off[t : t + 2] * self._class_weights
        return self.learned_weights_[lo:hi].reshape(-1, self._class_weights)

    @property
    def prior_weights_(self):
        """Weights of the class prior: one per class at ``weight_level`` 3
        and 4, a single one below."""
        return self._term_weights(0)[0]

    @property
    def feature_weights_(self):
        """Weights of the conditional of every attribute, a list with one
        array per attribute, of shape (n_keys, n_classes) at ``weight_level``
        3 and 4 and (n_keys, 1) below. The rows are the combinations that the
        level tells apart (values of the attribute, with its parents at level
        4) seen in training, a single row at level 1. An attribute the model
        ignores has no rows."""
        children = self._tables.children.tolist()
        empty = np.empty((0, self._class_weights))
        return [
            self._term_weights(1 + children.index(i)) if i in children else empty
            for i in range(self.n_features_in_)
        ]

    def weight_table(self, feature, as_frame=False):
        """
        Return the weights of the conditional of an attribute with the
        combinations they belong to.

        ``weight_level`` decides what a weight is shared by: the whole
        conditional (level 1, a single row), each value of the attribute (2
        and 3), or each value with each state of the parents (4); from level
        3 on there is a column per class. A Gaussian attribute has no values
        to tell apart, so its rows are the states of its parents alone. Only
        the combinations that occur in training have a weight.

        Parameters
        ----------
        feature : int or str
            The attribute, by its index or by its name in
            ``feature_names_in_``.

        as_frame : bool, default=False
            If True, return a pandas DataFrame indexed by the value of the
            attribute and the values of its parents, with a column per class
            (or a single column ``weight``).

        Returns
        -------
        table : :class:`~sklearn.utils.Bunch` or DataFrame
            With ``as_frame=False``, a dictionary-like object with:

            feature : int
                Index of the attribute.
            weights : ndarray of shape (n_rows, n_classes) or (n_rows, 1)
                The weights, ``feature_weights_[feature]``.
            feature_values : ndarray of shape (n_rows,) or None
                Value of the attribute in every row, when the level tells
                the values apart.
            parents : list of int
                Parents of the attribute, when the level tells their states
                apart (empty otherwise).
            parent_values : ndarray of shape (n_rows, n_parents)
                Values of those parents in every row.
            classes : ndarray of shape (n_classes,) or None
                Class of every column, when there is one per class.
        """
        feature, m = self._factor(feature)
        dims = _LEVEL_DIMS["factor"][self.weight_level]
        keys = self._weight_keys[self._key_off[1 + m] : self._key_off[2 + m]]
        kind, _, _, cards = self._layout()
        values = None
        if "value" in dims and kind[feature] == DISCRETE:
            keys, codes = np.divmod(keys, cards[feature])
            values = self._child_values(feature)[codes]
        parents, parent_values = [], np.empty((len(keys), 0), dtype=object)
        if "parents" in dims:
            parents, parent_values = self._parent_states(m, keys)
        table = Bunch(
            feature=feature,
            weights=self._term_weights(1 + m),
            feature_values=values,
            parents=parents,
            parent_values=parent_values,
            classes=self.classes_ if self._class_weights > 1 else None,
        )
        if not as_frame:
            return table
        index = ([] if values is None else [values]) + list(parent_values.T)
        names = [] if values is None else [self._feature_name(feature)]
        names += [self._feature_name(p) for p in parents]
        columns = ["weight"] if table.classes is None else table.classes
        return self._table_frame(table.weights, index, names, columns)

    def _term_keys(self, inputs):
        """Key of the weight of every term within the term, before the class:
        the parent key and the value of the child as the level asks (0 at
        level 1 and for the class prior), shape (n_samples, 1 + n_factors),
        from the :meth:`_evaluation_inputs` of the samples. A category or
        parent value never seen in training gets -1, a key no weight is
        trained for."""
        codes = inputs.codes
        dims = _LEVEL_DIMS["factor"][self.weight_level]
        t = self._tables
        kind, _, _, cards = self._layout()
        if {"parents", "value"} <= set(dims) and np.any(
            t.n_combos * cards[t.children].astype(float) >= 2**62
        ):
            raise ValueError(
                "The parent states and values of an attribute exceed a 64-bit"
                " index; use a lower weight_level, fewer parents or fewer bins."
            )
        keys = np.zeros((len(codes), 1 + len(t.children)), dtype=np.int64)
        for m, j in enumerate(t.children):
            unseen = np.zeros(len(codes), dtype=bool)
            n_parents = t.n_parents[m]
            if "parents" in dims and n_parents:
                p = t.parents[m, :n_parents]
                keys[:, 1 + m] = inputs.pcodes[:, p] @ t.strides[m, :n_parents]
                unseen |= inputs.punseen[:, p].any(axis=1)
            # The value of a Gaussian child is not discretized
            if "value" in dims and kind[j] == DISCRETE:
                keys[:, 1 + m] = keys[:, 1 + m] * cards[j] + codes[:, j]
                unseen |= codes[:, j] < 0
            keys[unseen, 1 + m] = -1
        return keys

    def _setup_weights(self, keys):
        """Lay out the weights on the term keys of the training samples: the
        sorted keys seen of every term, then the class when the level has
        it, innermost (the order of :class:`WeightedAnDE`)."""
        dims = _LEVEL_DIMS["factor"][self.weight_level]
        seen = [np.unique(keys[:, t]) for t in range(keys.shape[1])]
        self._class_weights = len(self.classes_) if "class" in dims else 1
        sizes = np.array([len(s) for s in seen], dtype=np.int64)
        self._key_off = np.concatenate([[0], np.cumsum(sizes)])
        self._weight_keys = np.concatenate(seen)
        self.n_weights_ = int(sizes.sum()) * self._class_weights + 1

    def _design(self, inputs):
        """Log-terms of every sample, class and term (the class log-prior,
        then the conditional of every factor), shape (n, n_classes, n_terms),
        and the index of the weight of each, from the
        :meth:`_evaluation_inputs` of the samples."""
        keys = self._term_keys(inputs)
        n_classes = len(self.classes_)
        with _threads(self.n_jobs):
            terms = conditional_scores(inputs, self._tables, n_classes, True)
        terms[:, :, 0] = self.backoff_.class_log_prior_

        # The weight of class 0 of every term; those of the other classes
        # follow it (see _Design in ande.py)
        n_seen = self.n_weights_ - 1
        index = np.empty(
            keys.shape, dtype=np.int32 if n_seen < 2**31 - n_classes else np.int64
        )
        for t in range(keys.shape[1]):
            seen = self._weight_keys[self._key_off[t] : self._key_off[t + 1]]
            pos = np.minimum(np.searchsorted(seen, keys[:, t]), len(seen) - 1)
            offset = self._key_off[t] * self._class_weights
            index[:, t] = np.where(
                seen[pos] == keys[:, t], offset + pos * self._class_weights, n_seen
            )
        return terms, index

    def _kernel_args(self, terms, index):
        """The design of :func:`_weighted_geometric_scores` with one
        component: the index, the class step, the values and the number of
        terms."""
        return (
            index.ravel(),
            int(self._class_weights > 1),
            terms.ravel(),
            terms.shape[2],
        )

    def _scores(self, weights, terms, index):
        """``sum_t w_t log theta_t`` for every sample and class."""
        n_samples, n_classes, _ = terms.shape
        args = self._kernel_args(terms, index)
        return _weighted_geometric_scores(
            *args,
            _pad_last(weights, args[1] * (n_classes - 1)),
            np.zeros((0, 0, 0)),
            np.ones((n_samples, 1), dtype=bool),
            n_classes,
        )

    def _nll_and_grad(self, weights, terms, index, Y):
        """Conditional negative log-likelihood of the one-hot targets ``Y``
        and its gradient with respect to the weights."""
        scores = self._scores(weights, terms, index)
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        dscores = np.exp(log_prob) - Y
        n_samples, n_classes, _ = terms.shape
        args = self._kernel_args(terms, index)
        grad = _weighted_geometric_gradient(
            *args,
            dscores,
            np.ones((n_samples, 1), dtype=bool),
            weights.size + args[1] * (n_classes - 1),
        )
        return -np.sum(log_prob * Y), _fold_padding(grad, weights.size)

    def _joint_log_likelihood(self, X):
        """Computes the weighted ``log P^w(y, x)`` for every sample and class,
        in row chunks that bound the memory of the log-terms.

        Returns
        -------
        jll : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, attributes=["classes_", "parents_", "learned_weights_"])
        X = _validate_input(self, X, reset=False)
        n_classes, n_terms = len(self.classes_), len(self._key_off) - 1
        jll = np.empty((X.shape[0], n_classes))
        for rows in _row_chunks(X.shape[0], n_classes * n_terms):
            terms, index = self._design(self._evaluation_inputs(X[rows]))
            jll[rows] = self._scores(self.learned_weights_, terms, index)
        return jll
