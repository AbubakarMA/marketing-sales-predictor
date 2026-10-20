
import numpy as np
from src.linear_regression import SimpleLinearRegression


def test_predict():
    model = SimpleLinearRegression()

    model.theta_0 = 5.0
    model.theta_1 = 2.0

    X = np.array([1, 2, 3])

    predictions = model.predict(X)

    expected = np.array([7, 9, 11])

    # Assert that predictions match expected values
    assert np.allclose(predictions, expected), f"Expected {expected}, but got {predictions}"

    
import pytest


def test_invalid_input():
    model = SimpleLinearRegression()

    X = np.array([1, 2, 3])
    y = np.array([10, 20])

    # Use pytest.raises(ValueError)
    # and call model.fit(X, y)
    with pytest.raises(ValueError):
        model.fit(X, y)