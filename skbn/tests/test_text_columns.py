"""Tests for the columns of labels (strings, pandas categories) taken by every
classifier, which must behave as their integer codes declared categorical."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from skbn import (
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

pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

CLASSIFIERS = [
    MixedNB(),
    AnDE(n_dependence=1),
    AnDE(n_dependence=2),
    WeightedAnDE(max_iter=5),
    AnJE(n_join=2),
    ALR(n_join=2, max_iter=5),
    GaussianAnJE(),
    GaussianALR(max_iter=5),
    TAN(),
    KDB(k=2),
    WeightedTAN(max_iter=5),
    WeightedKDB(k=2, max_iter=5),
]
COLOURS = np.array(["blue", "green", "red"])
SIZES = np.array(["L", "M", "S", "XL"])


def _data(n=150, seed=0):
    """A table with a float, a category, a string, a bool and a small integer
    column, and the same table with the labels replaced by their codes in
    sorted order (the encoding of the classifiers)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    size = (y + rng.integers(0, 2, n)) % 4
    colour = (y + (rng.random(n) < 0.3)) % 3
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n) + y,
            "size": pd.Categorical(SIZES[size], categories=["S", "M", "L", "XL"]),
            "colour": COLOURS[colour].astype(object),
            "flag": (y + rng.integers(0, 2, n)) % 2 == 0,
            "count": rng.integers(0, 4, n),
        }
    )
    codes = df.assign(size=size, colour=colour)
    return df, codes, y


def _params(model, **params):
    return clone(model).set_params(**params)


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_labels_are_their_codes(model):
    df, codes, y = _data()
    test, test_codes, _ = _data(60, seed=1)
    text = clone(model).fit(df, y)
    numeric = _params(model, categorical_features=[1, 2]).fit(codes, y)
    assert_array_equal(text.predict_proba(test), numeric.predict_proba(test_codes))
    assert_array_equal(text.predict(test), numeric.predict(test_codes))
    assert text.n_features_in_ == 5
    assert_array_equal(text.feature_names_in_, df.columns)


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_object_array_of_strings_and_numbers(model):
    df, codes, y = _data()
    X = df.assign(size=df["size"].astype(object)).to_numpy(dtype=object)
    X_codes = codes.to_numpy(dtype=float)
    text = clone(model).fit(X, y)
    numeric = _params(model, categorical_features=[1, 2]).fit(X_codes, y)
    assert_array_equal(text.predict_proba(X[:40]), numeric.predict_proba(X_codes[:40]))
    assert not hasattr(text, "feature_names_in_")

    # An array of strings only: every column holds labels
    X = X[:, 1:3].astype(str)
    text = clone(model).fit(X, y)
    numeric = _params(model, categorical_features=[0, 1]).fit(X_codes[:, 1:3], y)
    assert_array_equal(
        text.predict_proba(X[:40]), numeric.predict_proba(X_codes[:40, 1:3])
    )


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_category_with_many_integer_values_is_categorical(model):
    rng = np.random.default_rng(2)
    n = 300
    y = rng.integers(0, 2, n)
    values = (rng.integers(0, 15, n) + 15 * y) * 2  # 30 values, not 0..29
    x = rng.normal(size=n) + y
    X_cat = pd.DataFrame({"x": x, "v": pd.Categorical(values)})
    codes = np.searchsorted(np.unique(values), values)
    X_codes = np.column_stack([x, codes])
    text = clone(model).fit(X_cat, y)
    numeric = _params(model, categorical_features=[1]).fit(X_codes, y)
    assert_array_equal(text.predict_proba(X_cat), numeric.predict_proba(X_codes))

    # As plain integers, the same values are still typed continuous
    plain = MixedNB().fit(np.column_stack([x, values]), y)
    assert plain.feature_types_["gaussian"] == [0, 1]
    assert MixedNB().fit(X_cat, y).feature_types_["categorical"] == [1]


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_unseen_labels(model):
    df, codes, y = _data()
    test, test_codes, _ = _data(40, seed=3)
    test["colour"] = test["colour"].where(np.arange(40) % 3 > 0, "purple")
    test_codes["colour"] = test_codes["colour"].where(np.arange(40) % 3 > 0, 7)
    # A category of the dtype never seen in training is unseen as well
    test["size"] = test["size"].cat.add_categories("XXL")
    test.loc[::4, "size"] = "XXL"
    test_codes.loc[::4, "size"] = 9

    text = clone(model).fit(df, y)
    numeric = _params(model, categorical_features=[1, 2]).fit(codes, y)
    proba = text.predict_proba(test)
    assert_array_equal(proba, numeric.predict_proba(test_codes))
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_label_order_does_not_matter(model):
    df, _, y = _data()
    test, _, _ = _data(40, seed=4)
    reference = clone(model).fit(df, y).predict_proba(test)

    # The order of the categories of the dtype is not used at all
    reordered = df.assign(size=df["size"].cat.reorder_categories(["XL", "L", "S", "M"]))
    assert_array_equal(clone(model).fit(reordered, y).predict_proba(test), reference)

    # Renaming the labels changes their sorted order, not the model
    rename = {"blue": "zz", "green": "aa", "red": "mm"}
    sizes = {"S": "d", "M": "b", "L": "c", "XL": "a"}
    renamed = df.assign(
        colour=df["colour"].map(rename), size=df["size"].cat.rename_categories(sizes)
    )
    test_renamed = test.assign(
        colour=test["colour"].map(rename),
        size=test["size"].cat.rename_categories(sizes),
    )
    proba = clone(model).fit(renamed, y).predict_proba(test_renamed)
    assert_allclose(proba, reference, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_constant_text_column_is_ignored_as_a_constant_number(model):
    df, codes, y = _data()
    text = clone(model).fit(df.assign(const="same"), y)
    numeric = _params(model, categorical_features=[1, 2]).fit(codes.assign(const=0), y)
    assert_array_equal(
        text.predict_proba(df.assign(const="same")),
        numeric.predict_proba(codes.assign(const=0)),
    )
    # Another value at prediction is unseen, as another number would be
    assert_array_equal(
        text.predict_proba(df.assign(const="other")),
        numeric.predict_proba(codes.assign(const=1)),
    )


def test_declared_types_and_categories():
    df, codes, y = _data()
    model = MixedNB().fit(df, y)
    assert model.feature_types_ == {
        "gaussian": [0],
        "categorical": [1, 2, 4],
        "bernoulli": [3],
    }
    assert_array_equal(model.categories_[0], ["L", "M", "S", "XL"])
    assert_array_equal(model.categories_[1], COLOURS)
    assert_array_equal(model.categories_[2], [0, 1, 2, 3])
    # The constructor parameters are left as given
    assert model.categorical_features is None

    # A text column may be declared categorical or Bernoulli
    model = MixedNB(categorical_features=[2], bernoulli_features=[1]).fit(df, y)
    reference = MixedNB(categorical_features=[2], bernoulli_features=[1]).fit(codes, y)
    assert model.feature_types_ == reference.feature_types_
    assert_array_equal(model.predict_proba(df), reference.predict_proba(codes))


def test_category_and_strings_are_interchangeable_at_predict():
    df, _, y = _data()
    model = TAN().fit(df, y)
    swapped = df.assign(
        size=df["size"].astype(object), colour=df["colour"].astype("category")
    )
    assert_array_equal(model.predict_proba(swapped), model.predict_proba(df))


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_text_column_cannot_be_gaussian(model):
    df, _, y = _data()
    with pytest.raises(ValueError, match=r"features \[2\] hold labels .* gaussian"):
        _params(model, gaussian_features=[0, 2]).fit(df, y)


@pytest.mark.parametrize(
    "column",
    [
        pd.Series(["a", None, "b"] * 20, dtype=object),
        pd.Series(["a", np.nan, "b"] * 20, dtype=object),
        pd.Series(["a", None, "b"] * 20, dtype="string"),
        pd.Categorical(["a", None, "b"] * 20),
    ],
    ids=["None", "nan", "string", "category"],
)
def test_missing_labels(column):
    X = pd.DataFrame({"x": np.arange(60.0), "s": column})
    y = np.arange(60) % 2
    with pytest.raises(ValueError, match=r"(?s)missing values .* column 1.*impute"):
        AnDE().fit(X, y)
    complete = X.assign(s=["a", "c", "b"] * 20)
    model = AnDE().fit(complete, y)
    with pytest.raises(ValueError, match="missing values .* column 1"):
        model.predict(X)
    X = np.column_stack([np.arange(60.0), np.asarray(column, dtype=object)])
    with pytest.raises(ValueError, match="missing values .* column 1"):
        TAN().fit(X, y)


def test_feature_names_are_checked():
    df, _, y = _data()
    model = KDB(k=1).fit(df, y)
    with pytest.raises(ValueError, match="feature names should match"):
        model.predict(df[df.columns[::-1]])
    with pytest.raises(ValueError, match="feature names should match"):
        model.predict(df.rename(columns={"colour": "color"}))
    with pytest.warns(UserWarning, match="valid feature names"):
        with pytest.raises(ValueError, match="has 4 features"):
            model.predict(df.to_numpy(dtype=object)[:, :4])


def test_kind_of_columns_must_not_change():
    df, codes, y = _data()
    model = MixedNB().fit(df, y)
    with pytest.raises(ValueError, match=r"held them in fit, \[1, 2\]; got \[2\]"):
        model.predict(df.assign(size=codes["size"]))
    with pytest.raises(ValueError, match=r"held them in fit, \[1, 2\]; got \[\]"):
        model.predict(codes)
    numeric = MixedNB().fit(codes, y)
    with pytest.raises(ValueError, match="fitted on numeric columns only"):
        numeric.predict(df)


def test_objects_that_are_neither_strings_nor_numbers():
    X = np.array([["a", 1.0], ["b", 2.0], ["a", 0.5], ["b", 3.0]], dtype=object)
    y = [0, 1, 0, 1]
    for row, col in [(0, 0), (0, 1)]:
        bad = X.copy()
        bad[row, col] = {"foo": "bar"}
        with pytest.raises(TypeError, match="argument must be .* string.* number"):
            MixedNB().fit(bad, y)
    bad = X.copy()
    bad[0, 0] = 3
    with pytest.raises(TypeError, match="mixes strings with numbers"):
        MixedNB().fit(bad, y)

    # Numbers stored as objects are numbers, as scikit-learn takes them
    model = MixedNB().fit(X[:, [1]], y)
    assert model.feature_types_["gaussian"] == [0]


def test_list_input():
    X = [["a", 0.5], ["b", 2.0], ["a", 0.7], ["b", 3.0], ["c", 1.0]]
    y = [0, 1, 0, 1, 1]
    model = MixedNB().fit(X, y)
    reference = MixedNB(categorical_features=[0]).fit(
        [[0, 0.5], [1, 2.0], [0, 0.7], [1, 3.0], [2, 1.0]], y
    )
    assert_array_equal(
        model.predict_proba([["b", 1.5]]), reference.predict_proba([[1, 1.5]])
    )


def test_pipeline_and_model_selection():
    df, codes, y = _data()
    pipe = make_pipeline(FunctionTransformer(), TAN())
    scores = cross_val_score(pipe, df, y, cv=3)
    reference = cross_val_score(
        make_pipeline(FunctionTransformer(), TAN(categorical_features=[1, 2])),
        codes,
        y,
        cv=3,
    )
    assert_array_equal(scores, reference)

    search = GridSearchCV(pipe, {"tan__alpha": [0.1, 1.0]}, cv=3, n_jobs=1)
    search.fit(df, y)
    assert search.best_params_["tan__alpha"] in (0.1, 1.0)
    assert_array_equal(search.predict(df), search.best_estimator_.predict(df))


@pytest.mark.parametrize("model", CLASSIFIERS, ids=repr)
def test_clone_and_pickle(model):
    df, _, y = _data()
    test, _, _ = _data(40, seed=5)
    test.loc[0, "colour"] = "purple"
    fitted = clone(model).fit(df, y)
    proba = fitted.predict_proba(test)
    restored = pickle.loads(pickle.dumps(fitted))
    assert_array_equal(restored.predict_proba(test), proba)
    assert_array_equal(clone(fitted).fit(df, y).predict_proba(test), proba)
    assert not hasattr(clone(fitted), "_label_encoder")

    # Refitted on numbers, the model forgets the labels
    _, codes, _ = _data()
    refitted = fitted.fit(codes, y)
    assert refitted._label_encoder is None
