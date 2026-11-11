"""Discovery of the estimators of `skbn`, for the common tests."""

# Adapted from scikit-learn
# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings

_MODULE_TO_IGNORE = {"tests"}


def all_estimators():
    """Get a list of all estimators from `skbn`.

    This function crawls the package and gets all public classes defined in
    it that inherit from `BaseEstimator`. Classes that are defined in test
    modules are not included.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class, sorted by name.

    Examples
    --------
    >>> from skbn.tests._discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    """
    all_classes = []
    root = str(Path(__file__).parent.parent)  # skbn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="skbn."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            all_classes.extend(
                (name, cls) for name, cls in classes if not name.startswith("_")
            )

    # Keep the concrete estimators defined in skbn, not those it imports
    estimators = {
        (name, cls)
        for name, cls in all_classes
        if issubclass(cls, BaseEstimator)
        and not inspect.isabstract(cls)
        and cls.__module__.startswith("skbn")
    }
    # Sort for reproducibility; itemgetter keeps the sort off the classes
    return sorted(estimators, key=itemgetter(0))
