"""
.. raw:: html

    <h2>Hough Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal.windows import boxcar


def hough_spiker(signal: np.ndarray, window_length: int) -> np.ndarray:
    """
    Perform spike detection using the Hough Spiker Algorithm (HSA).

    This function detects spikes in an input signal by performing a progressive subtraction operation,
    where the signal is compared with a convolution result using a boxcar filter. If the signal value
    exceeds the filter result, the signal is modified by subtracting the filter, and a spike is recorded.

    Refer to the :ref:`hough_spiker_algorithm_desc` for a detailed explanation of the HSA.

    **Code Example**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.deconvolution import hough_spiker
        signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1])
        window_length = 3
        spikes = hough_spiker(signal, window_length)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import hough_spiker
        >>> signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1])
        >>> window_length = 3
        >>> spikes = hough_spiker(signal, window_length)
        >>> spikes
        array([0, 0, 1, 0, 0, 0, 0], dtype=int8)

    :param signal: The input signal to be analyzed. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param window_length: The length of the boxcar filter window.
    :type window_length: int
    :return: A 1D numpy array representing the detected spikes.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the window length is greater than the signal length.
    :raises TypeError: If the signal is not a numpy ndarray.

    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if window_length > len(signal):
        raise ValueError("Filter window size must be less than the length of the signal.")

    # Initialize the spike array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = boxcar(window_length)

    # Copy the signal for modification
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Iterate over the signal to detect spikes
    for t in range(len(signal_copy) - window_length + 1):
        # Count how many values match or exceed the filter window values
        match_count = np.sum(signal_copy[t : t + window_length] >= filter_window)

        # If all values match or exceed, a spike is detected
        if match_count == window_length:
            signal_copy[t : t + window_length] -= filter_window
            spikes[t] = 1

    return spikes
