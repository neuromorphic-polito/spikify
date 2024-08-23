"""Burst Encoding Algorithm."""
import numpy as np


def burst_encoding(signal: np.ndarray, n_max: int, t_min: int, t_max: int, length: int) -> np.ndarray:
    """
    Perform burst encoding on the input signal.

    Args:
        signal (np.ndarray): The input signal as a NumPy array.
        n_max (int): Maximum number of spikes.
        t_min (int): Minimum inter-spike interval (ISI).
        t_max (int): Maximum inter-spike interval (ISI).
        length (int): The total length of the encoding.

    Returns:
        np.ndarray: A 1D array representing the burst-encoded spike train.
    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if len(signal) % length != 0:
        raise ValueError(
            f"The burst_encoding length ({length}) is not a multiple of the signal length ({len(signal)})."
        )

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to length-sized chunks
    signal = np.mean(signal.reshape(-1, length), axis=1)

    # Normalize the signal
    signal_max = signal.max()
    if signal_max > 0:
        signal /= signal_max

    # Calculate spike numbers and ISI values
    spike_num = np.ceil(signal * n_max).astype(int)
    ISI = np.ceil(t_max - signal * (t_max - t_min)).astype(int)

    # Ensure the signal length can accommodate the spikes
    required_length = np.max(spike_num * (ISI + 1))
    if length < required_length:
        raise ValueError(f"Invalid stream length, the min length is {required_length}")

    # Initialize the spike array
    spikes = np.zeros((len(signal), length), dtype=int)

    # Populate the spike train based on ISI and spike number
    for i in range(len(signal)):
        spike_times = np.arange(0, spike_num[i] * (ISI[i] + 1), ISI[i] + 1)
        spikes[i, spike_times[:length]] = 1

    # Reshape the spike train into a 1D array
    return spikes.reshape(-1)
