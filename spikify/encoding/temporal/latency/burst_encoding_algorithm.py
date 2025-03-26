"""
.. raw:: html

    <h2>Burst Encoding Algorithm</h2>
"""

import numpy as np


def burst_encoding(signal: np.ndarray, n_max: int, t_min: int, t_max: int, length: int) -> np.ndarray:
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
    :type length: int
    :return: A 1D numpy array representing the burst-encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or the length is not a multiple of the signal length.
    :raises ValueError: If the required spike train length exceeds the provided length.

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
    spikes = np.zeros((len(signal), length), dtype=np.int8)

    # Populate the spike train based on ISI and spike number
    for i in range(len(signal)):
        spike_times = np.arange(0, spike_num[i] * (ISI[i] + 1), ISI[i] + 1)
        spikes[i, spike_times[:length]] = 1

    # Reshape the spike train into a 1D array
    return spikes.reshape(-1)
