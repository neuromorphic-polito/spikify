"""
.. raw:: html

    <h2>Moving Window Algorithm</h2>
"""

import numpy as np


def moving_window(signal: np.ndarray, window_length: int) -> np.ndarray:
    """
    Perform Moving Window encoding on the input signal.

    This function takes a continuous signal and converts it into a spike train using a moving window and
    threshold-based approach. A spike is generated when the signal exceeds the calculated `Base` plus or minus a
    specified `Threshold`.

    Refer to the :ref:`moving_window_algorithm_desc` for a detailed explanation of the Moving Window encoding
    algorithm.

    **Code Example:**

    .. code-block:: python

            import numpy as np
            from spikify.encoding.temporal.contrast import moving_window
            signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
            window_length = 3
            encoded_signal = moving_window(signal, window_length)
            encoded_signal

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.contrast import moving_window
        >>> signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        >>> window_length = 3
        >>> encoded_signal = moving_window(signal, window_length)
        >>> encoded_signal
        array([0, 0, 0, 1, 1, 1], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param window_length: The size of the sliding window for calculating the base mean.
    :type window_length: int
    :return: A 1D numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty.
    :raises ValueError: If the window length is greater than the length of the signal.
    :raises TypeError: If the signal is not a numpy ndarray.

    """

    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    variation = np.diff(signal[1:], prepend=signal[0])
    threshold = np.mean(np.abs(variation))

    spikes = np.zeros_like(signal, dtype=np.int8)

    # Compute the moving window mean and apply thresholds
    for t in range(len(signal)):
        base = np.mean(signal[:window_length]) if t < window_length else np.mean(signal[t - window_length : t])

        if signal[t] > base + threshold:
            spikes[t] = 1
        elif signal[t] < base - threshold:
            spikes[t] = -1

    return spikes
