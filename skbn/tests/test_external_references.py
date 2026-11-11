"""Agreement with independent, established implementations.

The references are outputs of other software, frozen in ``skbn/tests/data``
by ``tools/reference/make_reference.py`` (see ``tools/reference/README.md``
for the versions and how to regenerate them), so these tests need neither
Java nor the other packages:

- Weka 3.8.6 (``nz.ac.waikato.cms.weka:weka-stable:3.8.6``):
  ``weka.classifiers.bayes.NaiveBayes`` with its defaults;
  ``weka.classifiers.bayes.BayesNet -D -Q
  weka.classifiers.bayes.net.search.local.TAN -- -S ENTROPY
  -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A <alpha>``;
  ``weka.filters.supervised.attribute.Discretize`` with its defaults
  (Fayyad and Irani's MDL criterion, no ``-E``, no ``-K``);
- Weka's AODE package (``nz.ac.waikato.cms.weka:
  averagedOneDependenceEstimators:1.2.1``): ``weka.classifiers.bayes.AODE
  -F <limit>`` (Laplace estimates, no ``-M``);
- pgmpy 1.1.2 ``TreeSearch(estimator_type="tan")`` and pyAgrum 3.3.2
  ``pyagrum.skbn.BNClassifier(learningMethod="TAN", prior="NoPrior")``, for
  the structure of TAN.

The discrete datasets are integer-coded once in Python (quantile bins of
iris, wine and breast cancer, and a synthetic dataset with four unbalanced
classes and rare values) and fed unchanged to every implementation. Every
value of a test row occurs in training, and every value declared to Weka
occurs in training, since Weka's Laplace denominators count the declared
values and skbn's the values seen in training.

Where the models are the same, the posteriors are compared with
``atol=1e-13``: the largest difference measured is 2.1e-15 (a few ulp, Weka
multiplies probabilities where skbn adds logarithms), and the margin covers
summation order and libm differences across platforms. Two
conventions differ, and the tests account for them explicitly:

- the class prior: Weka smooths it, ``(n_c + a) / (n + K a)``, while skbn
  uses the maximum likelihood ``n_c / n`` of scikit-learn's naive Bayes. As
  a constant factor per class, it is swapped for Weka's in the posteriors of
  skbn before comparing (`_with_weka_prior`). AODE does not use it, except
  when it falls back to naive Bayes.
- numeric attributes in Weka's NaiveBayes, see `test_naive_bayes_numeric`.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from skbn import TAN, AnDE, MDLPDiscretizer, MixedNB

DATA = Path(__file__).parent / "data"
ATOL = 1e-13
NOMINAL = ["iris_q3", "wine_q4", "breast_cancer_q3", "synthetic"]


def _load(name):
    path = DATA / name
    if not path.is_file():
        pytest.skip(
            f"Reference fixture {name} not found in {DATA}: the test data were"
            " not installed with the package (see tools/reference/README.md)."
        )
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def datasets():
    return _load("reference_datasets.json")["datasets"]


@pytest.fixture(scope="module")
def weka():
    return _load("reference_weka.json")


@pytest.fixture(scope="module")
def tan_structure():
    return _load("reference_tan_structure.json")["tan"]


def _split(entry):
    return tuple(
        np.asarray(entry[k]) for k in ("X_train", "y_train", "X_test", "y_test")
    )


def _with_weka_prior(proba, y_train, alpha):
    """Posteriors with the class prior n_c / n of skbn replaced by the
    smoothed (n_c + alpha) / (n + K alpha) of Weka."""
    k = proba.shape[1]
    n_c = np.bincount(y_train, minlength=k)
    ratio = (n_c + alpha) / (len(y_train) + k * alpha) / (n_c / len(y_train))
    scaled = proba * ratio
    return scaled / scaled.sum(axis=1, keepdims=True)


def _edges(parents):
    return sorted(sorted([i, int(p)]) for i, p in enumerate(parents) if p >= 0)


# --- Naive Bayes --------------------------------------------------------------


@pytest.mark.parametrize("name", NOMINAL)
def test_naive_bayes_nominal(datasets, weka, name):
    # Weka NaiveBayes: (N(c, v) + 1) / (N(c) + r) per attribute, the
    # estimate of MixedNB(alpha=1) for categorical features
    X_train, y_train, X_test, _ = _split(datasets[name])
    model = MixedNB(categorical_features=list(range(X_train.shape[1])))
    proba = model.fit(X_train, y_train).predict_proba(X_test)
    expected = np.asarray(weka["naive_bayes"][name]["proba"])
    assert_allclose(_with_weka_prior(proba, y_train, 1.0), expected, rtol=0, atol=ATOL)
    if name == "iris_q3":
        # Equal class counts: both priors are uniform
        assert_allclose(proba, expected, rtol=0, atol=ATOL)


def _weka_precision(X):
    """Weka's precision of every numeric attribute: the mean gap between
    consecutive distinct training values."""
    return np.array([np.diff(np.unique(col)).mean() for col in X.T])


@pytest.mark.parametrize("name", ["gaussian_grid", "iris_numeric"])
def test_naive_bayes_numeric(datasets, weka, name):
    # Weka's NormalEstimator (weka/estimators/NormalEstimator.java) differs
    # from a Gaussian density in two ways:
    # 1. values are rounded to a precision p, the mean gap between the
    #    distinct training values of the attribute, before the (population)
    #    mean and standard deviation are computed;
    # 2. the likelihood of x is the probability of [x - p/2, x + p/2],
    #    (Phi((x - mu + p/2) / sd) - Phi((x - mu - p/2) / sd)) / p,
    #    not the density.
    # skbn fitted on Weka's rounded values gives Weka's mean and standard
    # deviation, and Weka's posteriors follow from them with 2.
    X_train, y_train, X_test, _ = _split(datasets[name])
    ref = weka["naive_bayes"][name]
    normal = np.asarray(ref["normal"])
    att, cls = normal[:, 0].astype(int), normal[:, 1].astype(int)
    mean, sd, precision = normal[:, 2:].T

    d = X_train.shape[1]
    assert_allclose(precision, _weka_precision(X_train)[att], rtol=1e-13)
    # Weka's own value of p, as rounding is sensitive to its last bit
    p = np.zeros(d)
    p[att] = precision
    X_round = np.rint(X_train / p) * p
    model = MixedNB(gaussian_features=list(range(d)), var_smoothing=0)
    model.fit(X_round, y_train)
    est = model.estimators_["gaussian"]
    assert_allclose(est.theta_[cls, att], mean, rtol=1e-13)
    assert_allclose(np.sqrt(est.var_[cls, att]), sd, rtol=1e-13)
    # Weka's floor p / 6 on the standard deviation is not reached
    assert np.all(sd > p[att] / 6)

    x = np.rint(X_test / p) * p
    mu, s = est.theta_[None], np.sqrt(est.var_)[None]
    upper = norm.cdf((x[:, None, :] - mu + p / 2) / s)
    lower = norm.cdf((x[:, None, :] - mu - p / 2) / s)
    joint = np.prod((upper - lower) / p, axis=2) * np.exp(model.class_log_prior_)
    proba = _with_weka_prior(joint / joint.sum(axis=1, keepdims=True), y_train, 1.0)
    expected = np.asarray(ref["proba"])
    assert_allclose(proba, expected, rtol=0, atol=ATOL)

    # MixedNB itself, on the original values, only predicts the same labels:
    # the largest posterior differences are 1.6e-3 on the integer grid
    # (where p = 1 and rounding changes nothing) and 0.13 on iris
    plain = MixedNB(gaussian_features=list(range(d)), var_smoothing=0)
    plain_proba = plain.fit(X_train, y_train).predict_proba(X_test)
    assert_array_equal(plain_proba.argmax(axis=1), expected.argmax(axis=1))
    if name == "gaussian_grid":
        assert_array_equal(X_round, X_train)
        assert (
            np.abs(_with_weka_prior(plain_proba, y_train, 1.0) - expected).max() < 2e-3
        )


# --- TAN ----------------------------------------------------------------------


@pytest.mark.parametrize("name", NOMINAL)
def test_tan_structure(datasets, weka, tan_structure, name):
    X_train, y_train, _, _ = _split(datasets[name])
    d = X_train.shape[1]
    model = TAN(categorical_features=list(range(d))).fit(X_train, y_train)
    edges = _edges(model.parents_)
    other = tan_structure[name]

    # Weka (-S ENTROPY: the score gain of a parent is N * I(A_i; A_j | C))
    # and pgmpy build the same maximum spanning tree
    assert edges == _edges(weka["tan"][name]["parents"])
    assert edges == other["pgmpy_edges"]
    assert_allclose(model.conditional_mutual_info_, other["pgmpy_cmi"], atol=ATOL)

    # pyAgrum's TAN adds the edges in decreasing CMI while they do not close a
    # cycle and their CMI is at least sum_{i<j} CMI_ij / (d (d - 1)), so it
    # keeps the edges of the tree above that threshold (here all of them)
    cmi = model.conditional_mutual_info_
    threshold = cmi[np.triu_indices(d, 1)].sum() / (d * (d - 1))
    assert other["pyagrum_edges"] == [e for e in edges if cmi[e[0], e[1]] >= threshold]


@pytest.mark.parametrize("alpha", ["1.0", "0.5"])
@pytest.mark.parametrize("name", NOMINAL)
def test_tan_posteriors(datasets, weka, name, alpha):
    # Weka's SimpleEstimator: (N(x_i, pa_i) + A) / (N(pa_i) + r_i A) for every
    # node, the estimate of skbn's alpha, including the uniform 1 / r_i of an
    # augmented state never seen. Weka roots the tree at an end of its
    # strongest edge; with smoothing, the posteriors depend on the root, so
    # skbn is given the same one.
    X_train, y_train, X_test, _ = _split(datasets[name])
    ref = weka["tan"][name]
    parents = np.asarray(ref["parents"])
    root = int(np.flatnonzero(parents < 0)[0])
    model = TAN(
        alpha=float(alpha),
        root=root,
        categorical_features=list(range(X_train.shape[1])),
    ).fit(X_train, y_train)
    assert_array_equal(model.parents_, parents)
    proba = _with_weka_prior(model.predict_proba(X_test), y_train, float(alpha))
    assert_allclose(proba, ref["proba"][alpha], rtol=0, atol=ATOL)


# --- AODE ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, limit",
    [(name, "1") for name in NOMINAL] + [("synthetic", "100"), ("synthetic", "150")],
)
def test_aode_posteriors(datasets, weka, name, limit):
    # Weka's AODE without -M: P(y, x_p) = (N(y, x_p) + 1) / (n + K r_p) and
    # P(x_i | y, x_p) = (N(y, x_p, x_i) + 1) / (N(y, x_p) + r_i), averaged
    # over the parents whose value occurs at least F times (-F); the
    # estimates of AnDE(n_dependence=1, alpha=1, min_parent_count=F). With
    # no such parent both fall back to naive Bayes, whose class prior differs
    # (see the module docstring).
    X_train, y_train, X_test, _ = _split(datasets[name])
    d = X_train.shape[1]
    model = AnDE(
        n_dependence=1,
        alpha=1.0,
        min_parent_count=int(limit),
        categorical_features=list(range(d)),
    ).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    expected = np.asarray(weka["aode"][name][limit]["proba"])

    counts = [np.bincount(X_train[:, j]) for j in range(d)]
    freq = np.column_stack([counts[j][X_test[:, j]] for j in range(d)])
    fallback = ~(freq >= int(limit)).any(axis=1)
    if limit == "150":
        assert fallback.any() and not fallback.all()
    assert_allclose(proba[~fallback], expected[~fallback], rtol=0, atol=ATOL)
    assert_allclose(
        _with_weka_prior(proba[fallback], y_train, 1.0),
        expected[fallback],
        rtol=0,
        atol=ATOL,
    )


# --- MDL discretization -------------------------------------------------------


@pytest.mark.parametrize(
    "name, loader",
    [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ],
)
def test_mdlp_cut_points(weka, name, loader):
    # Weka's supervised Discretize (Fayyad and Irani, defaults) on the full
    # dataset; both place a cut midway between consecutive distinct values,
    # (a + b) / 2, so the cut-points are compared exactly
    X, y = loader(return_X_y=True)
    model = MDLPDiscretizer().fit(X, y)
    expected = weka["mdl"][name]["cut_points"]
    assert len(model.cut_points_) == len(expected)
    for cuts, ref in zip(model.cut_points_, expected):
        assert_array_equal(cuts, np.asarray(ref, dtype=float))
