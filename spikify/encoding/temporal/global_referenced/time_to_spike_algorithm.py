"""TimeToFirstSpike  Algorithm."""

import numpy as np


def time_to_first_spike(signal: np.array, interval: int) -> np.ndarray:
    """
    Perform time-to-first-spike encoding on the input signal.

    Parameters:
    - signal: Input signal as a numpy array.
    - interval: Interval size for encoding.

    Returns:
     - spikes: A numpy array of the same shape as the signal, with 1 indicating a detected spike, 0 otherwise.
    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if len(signal) % interval != 0:
        raise ValueError(
            f"The time_to_spike interval ({interval}) is not a multiple of the signal length ({len(signal)})."
        )

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to interval-sized chunks
    signal = np.mean(signal.reshape(-1, interval), axis=1)

    # Normalize the signal
    signal_max = signal.max()
    if signal_max > 0:
        signal /= signal_max

    # Calculate intensity based on the signal
    with np.errstate(divide="ignore"):  # Avoid division warnings
        intensity = np.where(signal > 0, 0.1 * np.log(1 / signal), 2)

    # Create bins and quantize the intensity
    bins = np.linspace(0, 1, interval)
    levels = np.searchsorted(bins, intensity)

    # Create the spike matrix and set spikes
    spike = np.zeros((signal.shape[0], interval), dtype=np.int8)
    spike[np.arange(signal.shape[0]), np.clip(levels, 0, interval - 1)] = 1

    # Reshape the spike array into 1D
    return spike.reshape(-1)
