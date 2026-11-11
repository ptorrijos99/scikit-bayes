"""Versions of the environment and a check of the installation."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import platform
import sys
import time
import warnings

import numpy as np


def show_versions():
    """Print the versions of scikit-bayes and of the software it runs on.

    Paste the output in a bug report.

    Examples
    --------
    >>> import skbn
    >>> skbn.show_versions()  # doctest: +SKIP
    """
    import numba

    from . import __version__

    print(f"{'python':>14}: {sys.version.split()[0]} ({platform.platform()})")
    print(f"{'scikit-bayes':>14}: {__version__}")
    for name in (
        "numpy",
        "scipy",
        "sklearn",
        "numba",
        "llvmlite",
        "joblib",
        "threadpoolctl",
        "pandas",
    ):
        try:
            version = importlib.import_module(name).__version__
        except ImportError:
            version = "not installed"
        print(f"{name:>14}: {version}")
    print(
        f"{'numba threads':>14}: {numba.config.NUMBA_NUM_THREADS}"
        f" (threading layer: {numba.config.THREADING_LAYER})"
    )


def _models(n_jobs):
    """The classifiers the check runs, with few iterations for the weighted
    ones: it compiles their kernels, it does not tune them."""
    from . import (
        ALR,
        KDB,
        TAN,
        AnDE,
        AnJE,
        GaussianALR,
        GaussianAnJE,
        MixedNB,
        WeightedAnDE,
        WeightedKDB,
        WeightedTAN,
    )

    return {
        "MixedNB()": MixedNB(),
        "AnDE(n_dependence=1)": AnDE(n_dependence=1, n_jobs=n_jobs),
        "AnDE(n_dependence=2)": AnDE(n_dependence=2, n_jobs=n_jobs),
        "WeightedAnDE()": WeightedAnDE(max_iter=20, n_jobs=n_jobs),
        "AnJE()": AnJE(n_jobs=n_jobs),
        "ALR()": ALR(max_iter=20, l2_reg=1.0, n_jobs=n_jobs),
        "GaussianAnJE()": GaussianAnJE(n_jobs=n_jobs),
        "GaussianALR()": GaussianALR(max_iter=20, l2_reg=1.0, n_jobs=n_jobs),
        "TAN()": TAN(n_jobs=n_jobs),
        "KDB(k=2)": KDB(k=2, n_jobs=n_jobs),
        "WeightedTAN()": WeightedTAN(max_iter=20, n_jobs=n_jobs),
        "WeightedKDB(k=2)": WeightedKDB(k=2, max_iter=20, n_jobs=n_jobs),
    }


def check_installation(n_jobs=None):
    """Fit and evaluate every model once and report what it found.

    The first use of a model compiles its kernels, which takes from a second
    to some ten seconds per model and is cached on disk. Running this check
    after installing or upgrading pays that cost once, in one place, and
    verifies the results on the iris data: probabilities that are finite and
    sum to one, a training accuracy above 0.9, the equality of
    :class:`MixedNB` with scikit-learn's ``GaussianNB`` and the cut-points of
    :class:`MDLPDiscretizer`. It is what ``python -m skbn`` runs.

    Parameters
    ----------
    n_jobs : int, default=None
        Number of threads of the kernels. With more than one, the threaded
        versions of the kernels are the ones compiled.

    Returns
    -------
    ok : bool
        Whether every check passed.
    """
    from sklearn.datasets import load_iris
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.naive_bayes import GaussianNB

    from . import MDLPDiscretizer

    show_versions()
    X, y = load_iris(return_X_y=True)
    # A categorical and a binary attribute next to the continuous ones
    X = np.column_stack([X, np.digitize(X[:, 2], [2.5, 4.9]), X[:, 3] > 1.6])
    failed = []

    def report(name, seconds, problems):
        status = "ok" if not problems else "FAILED: " + "; ".join(problems)
        print(f"  {name:<24} {seconds:6.1f} s  {status}")
        if problems:
            failed.append(name)

    print("\nFirst use of every model (compiles its kernels):")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name, model in _models(n_jobs).items():
            start = time.perf_counter()
            problems = []
            try:
                proba = model.fit(X, y).predict_proba(X)
                if not np.all(np.isfinite(proba)):
                    problems.append("probabilities are not finite")
                elif not np.allclose(proba.sum(axis=1), 1.0):
                    problems.append("probabilities do not sum to one")
                elif np.mean(model.classes_[proba.argmax(axis=1)] == y) < 0.9:
                    problems.append("training accuracy below 0.9")
            except Exception as error:  # Reported, so that the check goes on
                problems.append(f"{type(error).__name__}: {error}")
            report(name, time.perf_counter() - start, problems)

        start = time.perf_counter()
        problems = []
        ours = _models(None)["MixedNB()"].fit(X[:, :4], y).predict_proba(X[:, :4])
        theirs = GaussianNB().fit(X[:, :4], y).predict_proba(X[:, :4])
        if not np.allclose(ours, theirs, rtol=0, atol=1e-12):
            problems.append("differs from GaussianNB")
        report("MixedNB = GaussianNB", time.perf_counter() - start, problems)

        start = time.perf_counter()
        problems = []
        cuts = MDLPDiscretizer().fit(X[:, :4], y).cut_points_
        if [len(c) for c in cuts] != [2, 2, 2, 2]:
            problems.append("unexpected cut-points on iris")
        report("MDLPDiscretizer", time.perf_counter() - start, problems)

    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}.")
    else:
        print(
            "\nAll checks passed. The compiled kernels are cached on disk: from"
            " now on the models start in a fraction of a second."
        )
    return not failed
