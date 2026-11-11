# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("scikit-bayes")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"
from ._check import check_installation, show_versions
from .ande import AnDE, WeightedAnDE
from .anje import ALR, AnJE, GaussianALR, GaussianAnJE
from .discretization import DecisionTreeDiscretizer, MDLPDiscretizer
from .kdb import KDB, WeightedKDB
from .mixed_nb import MixedNB
from .tan import TAN, WeightedTAN

__all__ = [
    "MixedNB",
    "AnDE",
    "AnJE",
    "ALR",
    "GaussianAnJE",
    "GaussianALR",
    "WeightedAnDE",
    "TAN",
    "KDB",
    "WeightedTAN",
    "WeightedKDB",
    "DecisionTreeDiscretizer",
    "MDLPDiscretizer",
    "check_installation",
    "show_versions",
    "__version__",
]
