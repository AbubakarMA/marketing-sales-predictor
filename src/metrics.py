import numpy as np


def mean_absolute_error(actual, predicted):
    # Convert both inputs to 1D NumPy arrays
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)

    # Ensure both arrays have the same length
    if len(actual) != len(predicted):
        raise ValueError("Lengths must match.")

    # Calculate absolute errors
    absolute_errors = np.abs(actual - predicted)

    # Return the mean absolute error
    return np.mean(absolute_errors)