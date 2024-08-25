"""Threshold-based Representation Algorithm."""

import numpy as np


def threshold_based_representation(signal: np.array, factor: float) -> np.array:
    """
    Generate a spike train based on a threshold of signal variations.

    Args:
    - settings (dict): Configuration settings (requires 'tbrFactor').
    - signal (ndarray): Input signal array.

    Returns:
    - spike (ndarray): Output spike train based on threshold crossing.
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    spike = np.zeros_like(signal, dtype=np.int8)

    # Compute the variation and threshold
    variation = np.diff(signal[1:], prepend=signal[0])
    threshold = np.mean(variation) + factor * np.std(variation)
    variation = np.insert(variation, 0, variation[1])

    # Apply threshold conditions
    spike[variation > threshold] = 1
    spike[variation < -threshold] = -1

    return spike
