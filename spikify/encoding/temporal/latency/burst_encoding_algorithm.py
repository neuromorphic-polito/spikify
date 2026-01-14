"""
.. raw:: html

    <h2>Burst Encoding Algorithm</h2>
"""

import numpy as np


def burst_encoding(signal: np.ndarray, n_max: int, t_min: int, t_max: int, interval_length: int) -> np.ndarray:
    """
    Perform burst encoding on the input signal.

    This function encodes the input signal by generating bursts of spikes
    based on the number of spikes and the inter-spike interval (ISI).
    The encoding process takes into account the maximum number of spikes (n_max),
    the minimum (t_min) and maximum (t_max) ISI, and the desired length of the encoded signal.

    Refer to the :ref:`burst_encoding_algorithm_desc` for a detailed explanation of the Burst Encoding Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.latency import burst_encoding
        np.random.seed(42)
        signal = np.random.rand(16)
        n_max = 4
        t_min = 2
        t_max = 6
        length = 16
        encoded_signal = burst_encoding(signal, n_max, t_min, t_max, length)


    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.latency import burst_encoding
        >>> np.random.seed(42)
        >>> signal = np.random.rand(16)
        >>> n_max = 4
        >>> t_min = 2
        >>> t_max = 6
        >>> length = 16
        >>> encoded_signal = burst_encoding(signal, n_max, t_min, t_max, length)
        >>> encoded_signal
        array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param n_max: The maximum number of spikes in each burst.
    :type n_max: int
    :param t_min: The minimum inter-spike interval (ISI).
    :type t_min: int
    :param t_max: The maximum inter-spike interval (ISI).
    :type t_max: int
    :param length: The total length of the encoded signal.
    :type interval_length: int
    :return: A 1D numpy array representing the burst-encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or the length is not a multiple of the signal length.
    :raises ValueError: If the required spike train length exceeds the provided length.

    """
    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    if T % interval_length != 0:
        raise ValueError(f"The interval_length ({interval_length}) is not a factor of the signal length ({T}).")

    signal_copy = signal.copy()

    signal_copy = np.clip(signal_copy, 0, None)
    signal_copy = np.mean(signal_copy.reshape(-1, interval_length, F), axis=1)

    signal_max = signal_copy.max(axis=0)
    signal_max[signal_max == 0] = 1
    signal_copy /= signal_max

    spike_num = np.ceil(signal_copy * n_max).astype(int)
    ISI = np.ceil(t_max - signal_copy * (t_max - t_min)).astype(int)

    required_length = np.max(spike_num * (ISI + 1))
    if interval_length < required_length:
        raise ValueError(f"Invalid stream length, the min length is {required_length}")

    spikes = np.zeros((T // interval_length, interval_length, F), dtype=np.int8)

    for i in range(signal_copy.shape[0]):
        for f in range(F):
            spike_times = np.arange(0, spike_num[i, f] * (ISI[i, f] + 1), ISI[i, f] + 1)
            spikes[i, spike_times[spike_times < interval_length], f] = 1

    spikes = spikes.reshape(-1, F)

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes
