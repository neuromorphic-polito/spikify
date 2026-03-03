"""
.. raw:: html

    <h2>Time To First Spike Algorithm</h2>
"""

import numpy as np


def time_to_first_spike(signal: np.ndarray, interval_length: int) -> np.ndarray:
    """
    Perform Time To First Spike (TTFS) encoding on the input signal.

    This function implements a sparse temporal coding scheme where
    input intensity is encoded in the **timing of the first spike** within a fixed
    time window (``interval_length``). Stronger inputs produce earlier spikes.

    The signal is divided into non-overlapping blocks of length ``interval_length``.
    For each block, the mean intensity is computed and mapped to a spike latency using
    a logarithmic function (approximating an exponential decaying threshold). A single
    spike is placed at the corresponding time step within the block.

    .. note::
        - TTFS requires normalized input signals between 0 and 1. If the input signal contains negative
          values, they are shifted to be non-negative and then normalized.

    Refer to the :ref:`time_to_first_spike_algorithm_desc`
    for a detailed explanation of the Time-to-First-Spike Encoding Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.global_referenced import time_to_first_spike
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        interval_length = 4
        encoded_signal = time_to_first_spike(signal, interval_length)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.global_referenced import time_to_first_spike
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        >>> interval_length = 4
        >>> encoded_signal = time_to_first_spike(signal, interval_length)
        >>> encoded_signal
        array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int8)

    :param signal: Input signal to encode (1D or 2D: timestamps × features).
    :type signal: numpy.ndarray
    :param interval_length: Length of each time window (block) for latency mapping.
                            Must evenly divide the signal length. Larger values give
                            coarser temporal resolution but allow longer latency range.
    :type interval_length: int
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If signal is empty or interval_length does not divide signal length.

    """

    # Check for empty signal
    if signal.shape[0] == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    if T % interval_length != 0:
        raise ValueError(f"The interval_length ({interval_length}) is not a factor of the signal length ({T}).")

    signal_copy = signal.copy()

    # Normalize signal if signal has negative values
    shift = signal_copy.min(axis=0)
    shift[shift > 0] = 0  # only shift if negative values are present
    signal_copy -= shift

    # Compute max amplitude per feature to be used for scaling if max amplitude is grater than 1
    max_amp = signal_copy.max(axis=0)

    # Find features that require scaling
    features_to_scale = np.where(max_amp > 1)[0]

    for f in features_to_scale:
        signal_copy[:, f] /= max_amp[f]

    # Compute mean over the signal reshaped to interval-sized chunks
    signal_copy = np.mean(signal_copy.reshape(T // interval_length, interval_length, F), axis=1)

    intensity = np.full_like(signal_copy, 2.0)
    mask = signal_copy > 0.0
    intensity[mask] = 0.1 * np.log(1 / signal_copy[mask])

    bins = np.linspace(0, 1, interval_length)
    levels = np.searchsorted(bins, intensity)

    spikes = np.zeros((T // interval_length, interval_length, F), dtype=np.int8)
    for f in range(F):
        spikes[np.arange(T // interval_length), np.clip(levels[:, f], 0, interval_length - 1), f] = 1

    spikes = spikes.reshape(T, F)

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes
