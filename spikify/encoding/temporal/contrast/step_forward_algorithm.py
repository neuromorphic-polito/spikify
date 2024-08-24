"""Step Forward Algorithm."""

import numpy as np


def step_forward(signal: np.array, threshold: float) -> np.array:
    """
    Generate a spike train based on step forward threshold logic.

    Args:
    - signal (ndarray): Input signal array.
    - threshold (int): Threshold value for spike generation.

    Returns:
    - spike (ndarray): Output spike train with 1/-1 values based on threshold step logic.
    """
    spike = np.zeros_like(signal, dtype=np.int8)

    # Base value initialized at the start of the signal
    base = signal[0]
    for t, value in enumerate(signal):
        if value > base + threshold:
            spike[t] = 1
            base += thresholds
        elif value < base - threshold:
            spike[t] = -1
            base -= threshold

    return spike
