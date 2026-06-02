import pickle

import numpy as np
import pytest
from sklearn.datasets import load_iris

from skbn import AnDE, DecisionTreeDiscretizer, MixedNB


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.mark.parametrize(
    "estimator", [AnDE(n_dependence=1), MixedNB(), DecisionTreeDiscretizer()]
)
def test_pickle_roundtrip(iris_data, estimator):
    """Test that estimators can be pickled and unpickled."""
    X, y = iris_data

    if hasattr(estimator, "predict"):
        estimator.fit(X, y)
        preds = estimator.predict(X)

        pickled = pickle.dumps(estimator)
        unpickled = pickle.loads(pickled)

        preds_unpickled = unpickled.predict(X)
        np.testing.assert_array_equal(preds, preds_unpickled)
    else:
        estimator.fit(X, y)
        transformed = estimator.transform(X)

        pickled = pickle.dumps(estimator)
        unpickled = pickle.loads(pickled)

        transformed_unpickled = unpickled.transform(X)
        np.testing.assert_array_equal(transformed, transformed_unpickled)
