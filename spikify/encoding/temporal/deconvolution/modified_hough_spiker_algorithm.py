"""Modified Hough Spiker Algorithm."""

import numpy as np

from scipy.signal.windows import boxcar


def modified_hough_spiker(signal: np.array, window_length: int, threshold: float) -> np.array:
    """
    Detect spikes in a signal using a modified Hough Spiker method.

    Parameters:
    - signal: Input signal as a numpy array.
    - window_length: The length of the boxcar filter window.
    - threshold: Threshold for detecting spikes. The function detects a spike when the error between the
                      signal and the filter window is less than or equal to this threshold.

    Returns:
    - spikes: A numpy array of the same shape as the signal, with 1 indicating a detected spike, 0 otherwise.

    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if window_length > len(signal):
        raise ValueError("Filter window size must be less than the length of the signal.")

    # Initialize the spikes array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = boxcar(window_length)

    # Copy the signal for modification
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Iterate over the signal to detect spikes
    for t in range(len(signal)):
        # Determine the end index for the current window
        end_index = min(t + window_length, len(signal))

        # Extract the relevant segment of the signal and the corresponding filter window
        signal_segment = signal_copy[t:end_index]
        filter_segment = filter_window[: end_index - t]

        # Calculate the error for this segment
        error = np.sum(np.maximum(filter_segment - signal_segment, 0))

        # If the cumulative error is within the threshold, a spike is detected
        if error <= threshold:
            signal_copy[t:end_index] -= filter_segment
            spikes[t] = 1

    return spikes
