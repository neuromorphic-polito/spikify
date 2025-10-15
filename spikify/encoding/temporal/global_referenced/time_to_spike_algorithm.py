"""
.. raw:: html

    <h2>Time To First Spike Algorithm</h2>
"""

import numpy as np


def time_to_first_spike(signal: np.ndarray, interval: int) -> np.ndarray:
    """
    Perform time-to-first-spike encoding on the input signal.

    This function encodes the input signal by computing the time to the first spike
    based on a dynamically decaying threshold, following an exponential function.
    The time to the first spike is determined by the value of the signal relative
    to this threshold.

    Refer to the :ref:`time_to_first_spike_algorithm_desc`
    for a detailed explanation of the Time-to-First-Spike Encoding Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.global_referenced import time_to_first_spike
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        interval = 4
        encoded_signal = time_to_first_spike(signal, interval)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.global_referenced import time_to_first_spike
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        >>> interval = 4
        >>> encoded_signal = time_to_first_spike(signal, interval)
        >>> encoded_signal
        array([1, 0, 0, 0, 0, 1, 0, 0], dtype=int8)

    :param signal: The input signal to be encoded.This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param interval: The size of the interval used for encoding.
    :type interval: int
    :return: A 1D numpy array representing the time-to-first-spike encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the interval is not a multiple of the signal length.

    """

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    S, F = signal.shape
    # Check for invalid inputs
    if signal.shape[0] == 0:
        raise ValueError("Signal cannot be empty.")

    if signal.shape[0] % interval != 0:
        raise ValueError(
            f"The time_to_spike interval ({interval}) is not a multiple of the signal length ({len(signal)})."
        )

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to interval-sized chunks
    signal = np.mean(signal.reshape(signal.shape[0] // interval, interval, signal.shape[1]), axis=1)

    signal_max = signal.max(axis=0)  # shape (F,)

    for feature in range(F):
        if signal_max[feature] > 0:
            signal[:, feature] /= signal_max[feature]

    # Calculate intensity based on the signal
    with np.errstate(divide="ignore"):  # Avoid division warnings
        intensity = np.where(signal > 0, 0.1 * np.log(1 / signal), 2)

    # Create bins and quantize the intensity
    bins = np.linspace(0, 1, interval)
    levels = np.searchsorted(bins, intensity)

    # Create the spike matrix and set spikes
    spike = np.zeros((signal.shape[0], interval, signal.shape[1]), dtype=np.int8)
    for feature in range(signal.shape[1]):
        spike[np.arange(signal.shape[0]), np.clip(levels[:, feature], 0, interval - 1), feature] = 1

    # Reshape the spike array into 1D
    if spike.shape[2] == 1:
        spike = spike.reshape(-1)
    else:
        spike = spike.reshape(spike.shape[0] * spike.shape[1], spike.shape[2])
    return spike
