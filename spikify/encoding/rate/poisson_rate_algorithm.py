"""Poisson Rate Algorithm."""

import numpy as np


def poisson_rate(signal: np.array, interval: int) -> np.array:
    """
    Perform Poisson rate encoding on the input signal.

    Args:
        signal (np.ndarray): The input signal as a NumPy array.
        interval (int): The interval size for encoding.

    Returns:
        np.ndarray: A 1D array of encoded spike data after Poisson rate encoding.
    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if len(signal) % interval != 0:
        raise ValueError(
            f"The poisson_rate interval ({interval}) is not a multiple of the signal length ({len(signal)})."
        )

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to interval-sized chunks
    signal = np.mean(signal.reshape(-1, interval), axis=1)

    # Normalize the signal
    signal_max = signal.max()
    if signal_max > 0:
        signal /= signal_max

    # Initialize the spike array
    spikes = np.zeros((signal.shape[0], interval), dtype=np.int8)

    # Create bins for Poisson rate encoding
    bins = np.linspace(0, 1, interval + 1)

    # Generate Poisson spike trains
    for i, rate in enumerate(signal):
        if rate > 0:
            ISI = [-np.log(1 - np.random.random()) / (rate * interval)] * interval  # Inter-spike intervals
            spike_times = np.searchsorted(bins, np.cumsum(ISI)) - 1  # Find spike times
            spike_times = spike_times[spike_times < interval]  # Clip times within interval
            spikes[i, spike_times] = 1

    # Flatten the 2D array into a 1D spike train
    return spikes.flatten()
