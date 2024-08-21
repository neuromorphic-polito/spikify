import numpy as np
from scipy.signal.windows import boxcar


def ben_spiker(signal: np.array, bsa_filter_window: int, bsa_threshold: float) -> np.ndarray:
    """
    Detect spikes in a signal based on a boxcar filter window and a specified threshold.

    Parameters:
    - signal: Input signal as a numpy array.
    - bsa_filter_window: The length of the boxcar filter window.
    - bsa_threshold: Threshold for spike detection.

    Returns:
    - spikes: A numpy array of the same shape as the signal, with 1 indicating a detected spike, 0 otherwise.
    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if bsa_filter_window >= len(signal):
        raise ValueError("Filter window size must be less than the length of the signal.")

    # Initialize the spike array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = np.array(boxcar(bsa_filter_window), dtype=np.int64)

    # Copy of the signal to avoid modifying the original input
    signal_copy = np.copy(signal)

    # Iterate over the signal to detect spikes
    for t in range(len(signal) - bsa_filter_window + 1):
        # Calculate errors using the filter window
        segment = signal_copy[t:t + bsa_filter_window]
        error1 = np.sum(np.abs(segment - filter_window))
        error2 = np.sum(np.abs(segment))

        # Update signal and spike array if a spike is detected
        if error1 <= (error2 - bsa_threshold):
            signal_copy[t:t + bsa_filter_window] -= filter_window
            spikes[t] = 1

    return spikes
