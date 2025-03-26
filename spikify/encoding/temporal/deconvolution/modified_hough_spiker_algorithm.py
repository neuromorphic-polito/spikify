"""
.. raw:: html

    <h2>Modified Hough Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal.windows import boxcar


def modified_hough_spiker(signal: np.ndarray, window_length: int, threshold: float) -> np.ndarray:
    """
    Detect spikes in a signal using the Modified Hough Spiker Algorithm.

    This function detects spikes in an input signal by incorporating a threshold-based
    error accumulation mechanism. The signal is compared with a convolution result
    using a boxcar filter, and the error is accumulated over time. If the error remains
    within a specified threshold, a spike is detected, and the signal is modified.

    Refer to the :ref:`modified_hough_spiker_algorithm_desc` for a detailed explanation of the Modified Hough Spiker
    Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.deconvolution import modified_hough_spiker
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        window_length = 3
        threshold = 0.5
        spikes = modified_hough_spiker(signal, window_length, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import modified_hough_spiker
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        >>> window_length = 3
        >>> threshold = 0.5
        >>> spikes = modified_hough_spiker(signal, window_length, threshold)
        >>> spikes
        array([0, 0, 0, 0, 0, 0, 0], dtype=int8)

    :param signal: The input signal to be analyzed. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param window_length: The length of the boxcar filter window.
    :type window_length: int
    :param threshold: The threshold value for error accumulation.
    :type threshold: float
    :return: A 1D numpy array representing the detected spikes.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the window length is greater than the signal length.
    :raises TypeError: If the signal is not a numpy ndarray.

    """
    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if window_length > len(signal):
        raise ValueError("Filter window size must be less than the length of the signal.")

    # Initialize the spikes array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = boxcar(window_length)

    # Copy the signal for modification
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Iterate over the signal to detect spikes
    for t in range(len(signal)):
        # Determine the end index for the current window
        end_index = min(t + window_length, len(signal))

        # Extract the relevant segment of the signal and the corresponding filter window
        signal_segment = signal_copy[t:end_index]
        filter_segment = filter_window[: end_index - t]

        # Calculate the error for this segment
        error = np.sum(np.maximum(filter_segment - signal_segment, 0))

        # If the cumulative error is within the threshold, a spike is detected
        if error <= threshold:
            signal_copy[t:end_index] -= filter_segment
            spikes[t] = 1

    return spikes
