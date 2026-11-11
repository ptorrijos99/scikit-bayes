"""Tests for the installation check and the report of versions."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import skbn
from skbn import _check
from skbn.__main__ import main
from skbn.tests._discovery import all_estimators


def test_show_versions_lists_the_dependencies(capsys):
    skbn.show_versions()
    out = capsys.readouterr().out
    for name in ("python", "scikit-bayes", "numpy", "scipy", "sklearn", "numba"):
        assert f"{name}:" in out
    assert skbn.__version__ in out


def test_check_runs_every_classifier():
    # A classifier added to the package must be added to the check
    checked = {type(model).__name__ for model in _check._models(None).values()}
    classifiers = {
        name for name, cls in all_estimators() if hasattr(cls, "predict_proba")
    }
    assert checked == classifiers


def test_check_passes_and_reports_every_model(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "All checks passed" in out
    assert out.count("  ok") == len(_check._models(None)) + 2
    assert "FAILED" not in out


def test_check_reports_a_wrong_result(capsys, monkeypatch):
    # A model whose probabilities are wrong makes the check fail, and the
    # others are still run
    class Broken(skbn.TAN):
        def predict_proba(self, X):
            return np.full((len(X), 3), np.nan)

    models = _check._models
    monkeypatch.setattr(
        _check, "_models", lambda n_jobs: {**models(n_jobs), "TAN()": Broken()}
    )
    assert main([]) == 1
    out = capsys.readouterr().out
    assert "FAILED: probabilities are not finite" in out
    assert "1 check(s) failed: TAN()." in out
    assert out.count("  ok") == len(models(None)) + 1
