"""Hough Spiker Algorithm."""

import numpy as np

from scipy.signal.windows import boxcar


def hough_spiker(signal: np.ndarray, hsa_filter_window: int) -> np.ndarray:
    """
    Detect spikes in a signal using the Hough Spiker method.

    Parameters:
    - signal: Input signal as a numpy array.
    - hsa_filter_window: The length of the boxcar filter window.

    Returns:
    - spikes: A numpy array of the same shape as the signal, with 1 indicating a detected spike, 0 otherwise.
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if hsa_filter_window >= len(signal):
        raise ValueError("Filter window size must be less than the length of the signal.")

    # Initialize the spike array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = np.array(boxcar(hsa_filter_window), dtype=np.int64)

    # Copy the signal for modification
    signal_copy = np.copy(signal)

    # Iterate over the signal to detect spikes
    for t in range(len(signal_copy) - hsa_filter_window + 1):
        # Count how many values match or exceed the filter window values
        match_count = np.sum(signal_copy[t:t + hsa_filter_window] >= filter_window)

        # If all values match or exceed, a spike is detected
        if match_count == hsa_filter_window:
            signal_copy[t:t + hsa_filter_window] -= filter_window
            spikes[t] = 1

    return spikes
