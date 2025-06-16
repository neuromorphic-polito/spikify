"""
.. raw:: html

    <h2>Bens Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal.windows import boxcar  # da sistemare a scelta


def bens_spiker(
    signal: np.ndarray, window_length: int | list[int], threshold: float | list[float]
) -> np.ndarray:  # window_length può cambiare
    """
    Perform spike detection using Bens Spiker Algorithm.

    This function detects spikes in an input signal based on the comparison of cumulative errors calculated over a
    segment of the signal, which is filtered using a boxcar window. A spike is detected if the cumulative error between
    the filtered signal and the raw signal is below a certain threshold.

    Refer to the :ref:`bens_spiker_algorithm_desc` for a detailed explanation of the Ben's Spiker algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.deconvolution import bens_spiker
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        window_length = 3
        threshold = 0.5
        spikes = bens_spiker(signal, window_length, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import bens_spiker
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        >>> window_length = 3
        >>> threshold = 0.5
        >>> spikes = bens_spiker(signal, window_length, threshold)
        >>> spikes
        array([0, 0, 1, 0, 0, 0, 0], dtype=int8)

    :param signal: The input signal to be analyzed. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param window_length: The length of the boxcar filter window.
    :type window_length: int | list[int]
    :param threshold: Threshold value used to detect spikes. Can be a float or a list/array of floats.
    :type threshold: float | list[float]
    :return: A 1D numpy array representing the detected spikes.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the window length is greater than the signal length.
    :raises TypeError: If the signal is not a numpy ndarray.

    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    S, F = signal.shape

    if isinstance(window_length, int):
        window_length = [window_length] * F

    if np.any(np.array(window_length) > S):
        raise ValueError("All filter window sizes must be less than the length of the signal.")

    if isinstance(threshold, float):
        threshold = [threshold] * F

    if len(threshold) != F:
        raise ValueError("Threshold must match the number of features in the signal.")

    # Initialize the spike array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = [boxcar(w) for w in window_length]

    # Copy of the signal to avoid modifying the original input
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Iterate over the signal to detect spikes
    for j in range(signal.shape[1]):
        for t in range(len(signal[:, j]) - window_length[j] + 1):
            # Calculate errors using the filter window
            segment = signal_copy[t : t + window_length[j], j]
            error1 = np.sum(np.abs(segment - filter_window[j]), axis=0)
            error2 = np.sum(np.abs(segment), axis=0)

            # Update signal and spike array if a spike is detected
            if error1 <= (error2 - threshold[j]):
                signal_copy[t : t + window_length[j], j] -= filter_window[j]
                spikes[t, j] = 1
    if spikes.shape[-1] == 1:
        spikes = spikes.flatten()

    return spikes
