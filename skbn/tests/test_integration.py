import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from skbn import AnDE, DecisionTreeDiscretizer, MixedNB


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_pipeline_integration(iris_data):
    """Test classifier works in sklearn Pipeline."""
    X, y = iris_data
    pipe = make_pipeline(DecisionTreeDiscretizer(), AnDE(n_dependence=1))

    # Should fit without error
    pipe.fit(X, y)

    # Predict should return valid shape
    preds = pipe.predict(X)
    assert preds.shape == (150,)

    # Test cross-validation
    scores = cross_val_score(pipe, X, y, cv=3)
    assert len(scores) == 3
    assert scores.mean() > 0.5  # Should learn something


def test_gridsearch_integration(iris_data):
    """Test classifier works with GridSearchCV."""
    X, y = iris_data

    param_grid = {"n_dependence": [0, 1], "n_bins": [3, 5]}

    gs = GridSearchCV(AnDE(), param_grid, cv=2)
    gs.fit(X, y)

    assert hasattr(gs, "best_params_")
    assert gs.best_params_["n_dependence"] in [0, 1]
    assert gs.best_params_["n_bins"] in [3, 5]


def test_mixednb_pipeline(iris_data):
    """Test MixedNB in a pipeline."""
    X, y = iris_data
    pipe = make_pipeline(
        DecisionTreeDiscretizer(), MixedNB(categorical_features=[0, 1, 2, 3])
    )
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (150,)
