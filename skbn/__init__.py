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
from .ande import AnDE, WeightedAnDE
from .anje import ALR, AnJE, GaussianALR, GaussianAnJE
from .discretization import DecisionTreeDiscretizer, MDLPDiscretizer
from .kdb import KDB
from .mixed_nb import MixedNB
from .tan import TAN

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
    "DecisionTreeDiscretizer",
    "MDLPDiscretizer",
    "__version__",
]
