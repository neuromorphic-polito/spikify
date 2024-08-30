"""Zero-crossing Step Forward Algorithm."""

import numpy as np


def zero_cross_step_forward(signal: np.array, threshold: int) -> np.array:
    """
    Generate a spike train where positive values crossing a threshold trigger spikes.

    Args:
    - settings (dict): Configuration settings (requires 'zcsfThreshold').
    - signal (ndarray): Input signal array.

    Returns:
    - spike (ndarray): Output spike train based on zero-crossing threshold.

    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    spike = np.zeros_like(signal, dtype=np.int8)

    # Zero out negative values
    signal = np.maximum(signal, 0)

    # Apply threshold condition
    spike[signal > threshold] = 1

    return spike
