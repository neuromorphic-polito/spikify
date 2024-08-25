"""Moving Window Algorithm."""

import numpy as np


def moving_window(signal: np.array, window_length: int, threshold: float) -> np.array:
    """
    Generate a spike train using a moving window for threshold comparison.

    Args:
    - signal (ndarray): Input signal array.
    - window_length (int): Length of the moving window.
    - threshold (float): Threshold value for spike generation

    Returns:
    - spike (ndarray): Output spike train with 1/-1 values based on moving window logic.
    """
    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    spikes = np.zeros_like(signal, dtype=np.int8)

    # Compute the moving window mean and apply thresholds
    for t in range(len(signal)):
        base = np.mean(signal[:window_length]) if t < window_length else np.mean(signal[t - window_length : t])

        if signal[t] > base + threshold:
            spikes[t] = 1
        elif signal[t] < base - threshold:
            spikes[t] = -1

    return spikes
