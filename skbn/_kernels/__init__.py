"""Compiled kernels of the estimators, compiled with Numba on first use and
cached on disk.

- :mod:`._runtime`: the dispatch to the serial or the parallel version of a
  kernel, the constants and the scalar helpers that all the kernels share;
- :mod:`._states`: the augmented states ``Y* = (y, x_p)`` of the training
  rows, shared by the SPnDEs and the conditionals of TAN and KDB, and the
  samples to evaluate (:class:`EvaluationInputs`);
- :mod:`._spnde`: the counts and scores of the SPnDEs of the AnDE family
  (:class:`SPnDETables`);
- :mod:`._joins`: the n-joins of AnJE and ALR;
- :mod:`._weighted`: the scores and gradients of the discriminative weights;
- :mod:`._conditional`: the conditionals of TAN and KDB
  (:class:`ConditionalTables`);
- :mod:`._encoding`: the encoding of the input columns and the sufficient
  statistics of MixedNB.

The evaluation kernels take the named tuples themselves and read their
fields by name, so the order of the fields does not matter.

The estimators import the kernels from this package, which re-exports them.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

from ._conditional import (
    ConditionalTables,
    conditional_mutual_info,
    conditional_scores,
    count_conditionals,
    count_states,
)
from ._encoding import (
    _categorical_counts,
    _class_moments,
    _column_profiles,
    _column_ranges,
    _integer_codes,
    _table_codes,
)
from ._joins import (
    join_cells_seen,
    join_counts,
    join_key_counts,
    join_keys,
    join_objective,
    join_rows,
    join_scores,
)
from ._runtime import (
    DISCRETE,
    GAUSSIAN,
    IGNORED,
    LOG_ZERO,
    _Kernel,
    _threads,
    gaussian_log_norms,
)
from ._spnde import (
    _MAX_LIGHT,
    SPnDETables,
    component_scores,
    count_pairs,
    count_tables,
    count_triples,
    dense_estimates,
    index_states,
    pooled_scores,
)
from ._states import EvaluationInputs, _state_index
from ._weighted import (
    _pairwise_sum,
    _weighted_arithmetic_gradient,
    _weighted_arithmetic_scores,
    _weighted_geometric_gradient,
    _weighted_geometric_scores,
)

__all__ = [
    "DISCRETE",
    "GAUSSIAN",
    "IGNORED",
    "LOG_ZERO",
    "_MAX_LIGHT",
    "ConditionalTables",
    "EvaluationInputs",
    "SPnDETables",
    "_Kernel",
    "_categorical_counts",
    "_class_moments",
    "_column_profiles",
    "_column_ranges",
    "_integer_codes",
    "_pairwise_sum",
    "_state_index",
    "_table_codes",
    "_threads",
    "_weighted_arithmetic_gradient",
    "_weighted_arithmetic_scores",
    "_weighted_geometric_gradient",
    "_weighted_geometric_scores",
    "component_scores",
    "conditional_mutual_info",
    "conditional_scores",
    "count_conditionals",
    "count_pairs",
    "count_states",
    "count_tables",
    "count_triples",
    "dense_estimates",
    "gaussian_log_norms",
    "index_states",
    "join_cells_seen",
    "join_counts",
    "join_key_counts",
    "join_keys",
    "join_objective",
    "join_rows",
    "join_scores",
    "pooled_scores",
]
