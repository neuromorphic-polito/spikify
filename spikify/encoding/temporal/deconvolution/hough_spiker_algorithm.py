"""Hough Spiker Algorithm."""

import numpy as np

from scipy.signal.windows import boxcar


def hough_spiker(signal: np.array, window_length: int) -> np.array:
    """
    Detect spikes in a signal using the Hough Spiker method.

    Parameters:
    - signal: Input signal as a numpy array.
    - window_length: The length of the boxcar filter window.

    Returns:
    - spikes: A numpy array of the same shape as the signal, with 1 indicating a detected spike, 0 otherwise.
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if window_length > len(signal):
        raise ValueError("Filter window size must be less than the length of the signal.")

    # Initialize the spike array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = boxcar(window_length)

    # Copy the signal for modification
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Iterate over the signal to detect spikes
    for t in range(len(signal_copy) - window_length + 1):
        # Count how many values match or exceed the filter window values
        match_count = np.sum(signal_copy[t : t + window_length] >= filter_window)

        # If all values match or exceed, a spike is detected
        if match_count == window_length:
            signal_copy[t : t + window_length] -= filter_window
            spikes[t] = 1

    return spikes
