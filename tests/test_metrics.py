import numpy as np
import pytest

from src.metrics import mean_absolute_error


def test_mae_handles_broadcasting_bug():
    """Test actual shape (3,) against prediction shape (3, 1)."""

    actual = np.array([100, 200, 300])
    predicted = np.array([[110], [180], [330]])

    result = mean_absolute_error(actual, predicted)

    assert result == pytest.approx(20.0)


def test_mae_raises_error_for_unequal_lengths():
    """Test that unequal array lengths raise ValueError."""

    actual = np.array([100, 200, 300])
    predicted = np.array([110, 180])

    with pytest.raises(ValueError, match=r"Lengths must match\."):
        mean_absolute_error(actual, predicted)