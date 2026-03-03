"""
.. raw:: html

    <h2>Moving Window Algorithm</h2>
"""

import numpy as np


def moving_window(
    signal: np.ndarray, window_length: int, threshold: float | int | list[float | int] | np.ndarray
) -> np.ndarray:
    """
    Perform Moving Window (MW) encoding on the input signal.

    This function takes a continuous signal and converts it into a spike train using a moving window and
    threshold-based approach. A spike is generated when the signal exceeds the calculated `base` plus or minus a
    specified `threshold`.

    Refer to the :ref:`moving_window_algorithm_desc` for a detailed explanation of the Moving Window encoding
    algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.contrast import moving_window
        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        window_length = 3
        threshold = 0.2
        encoded_signal = moving_window(signal, window_length, threshold)
        encoded_signal

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.contrast import moving_window
        >>> signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        >>> window_length = 3
        >>> threshold = 0.2
        >>> encoded_signal = moving_window(signal, window_length, threshold)
        >>> encoded_signal
        array([0, 0, 0, 1, 1, 1], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param window_length: The size of the sliding window for calculating the signal base mean.
    :type window_length: int
    :param threshold: Threshold(s) for spike generation; scalar or 1D sequence matching features.
    :type threshold: float | int | list[float | int] | numpy.ndarray
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the threshold dimensions do not match the signal features.
    :raises TypeError: If the threshold parameter is of invalid dimension.

    """

    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    # Handle threshold
    if np.isscalar(threshold):
        thresholds = np.full(F, float(threshold))
    else:
        thresholds = np.asarray(threshold, dtype=float)
        if thresholds.ndim != 1:
            raise TypeError("Threshold must be a scalar or a 1D sequence of numbers.")
        if thresholds.size != F:
            raise ValueError("Threshold must match the number of features in the signal.")

    spikes = np.zeros_like(signal, dtype=np.int8)

    # First loop: t = 0 : window_length
    # For the first window_length samples, use the mean of available samples as base signal otherwise
    # the first window_length samples will not be encoded since there are not enough samples to fill the window
    for f in range(F):
        base = np.mean(signal[:window_length, f])
        for t in range(window_length):
            if signal[t, f] > base + thresholds[f]:
                spikes[t, f] = 1
            elif signal[t, f] < base - thresholds[f]:
                spikes[t, f] = -1

    # Second loop: t = window_length : T
    # For the rest of the signal, use the moving window to calculate the base signal
    for f in range(F):
        for t in range(window_length, T):
            base = np.mean(signal[t - window_length : t, f])
            if signal[t, f] > base + thresholds[f]:
                spikes[t, f] = 1
            elif signal[t, f] < base - thresholds[f]:
                spikes[t, f] = -1

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes
