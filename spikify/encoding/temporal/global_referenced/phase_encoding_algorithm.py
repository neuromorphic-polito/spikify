"""Phase Encoding Algorithm."""

import numpy as np


def phase_encoding(signal: np.array, num_bits: int) -> np.array:
    """
    Perform phase encoding on the input signal based on the given settings.

    Parameters:
    - signal: Input signal as a numpy array.
    - num_bits: Numer of bits for encoding.

    Returns:
     - spikes: A numpy array of the same shape as the signal, with 1 indicating a detected spike, 0 otherwise.
    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to bit-sized chunks
    signal = np.mean(signal.reshape(-1, num_bits), axis=1)

    # Normalize the signal if the maximum is greater than 0
    signal_max = signal.max()
    if signal_max > 0:
        signal /= signal_max

    # Compute the phase angles based on the signal
    phase = np.arcsin(signal)

    # Create phase bins and quantize the phase
    bins = np.linspace(0, np.pi / 2, 2**num_bits + 1)
    levels = np.searchsorted(bins, phase)

    # Adjust levels to avoid out-of-range values
    levels = np.clip(levels, 0, 2**num_bits - 1)

    # Convert levels to binary and flatten the result to a 1D spike array
    spikes = np.unpackbits(levels[:, None].astype(np.uint8), axis=1, bitorder="big")[:, -num_bits:].reshape(-1)

    return spikes
