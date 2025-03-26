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
