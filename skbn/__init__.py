# Authors: scikit-bayes developers
# License: BSD 3 clause

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("scikit-bayes")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"
from .ande import ALR, AnDE, AnJE, WeightedAnDE
from .mixed_nb import MixedNB
from .discretization import DecisionTreeDiscretizer

__all__ = ["MixedNB", "AnDE", "AnJE", "ALR", "WeightedAnDE", "DecisionTreeDiscretizer", "__version__"]
