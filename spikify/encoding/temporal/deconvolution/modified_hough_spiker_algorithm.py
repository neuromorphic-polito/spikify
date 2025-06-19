"""
.. raw:: html

    <h2>Modified Hough Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal.windows import get_window
from .utils import WindowType


def modified_hough_spiker(
    signal: np.ndarray,
    window_length: int | list[int],
    threshold: float | list[float],
    window_type: WindowType = "boxcar",
) -> np.ndarray:
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
    :param window_length: The length of the boxcar filter window. Can be a int or a list of ints.
    :type window_length: int | list[int]
    :param threshold: The threshold value for error accumulation. Can be a float or a list/array of floats.
    :type threshold: float or list of float
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

    if isinstance(threshold, float):
        thresholds = [threshold] * F
    elif isinstance(threshold, list):
        if not all(isinstance(w, float) for w in threshold):
            raise TypeError("All elements in threshold list must be float.")
        thresholds = threshold
    else:
        raise TypeError("Threshold must be a float or a list of floats.")

    if len(thresholds) != F:
        raise ValueError("Thresholds must match the number of features in the signal.")

    if isinstance(window_length, int):
        window_lengths = [window_length] * F
    elif isinstance(window_length, list):
        window_lengths = window_length
    else:
        raise TypeError("Window length must be an int or a list of ints.")

    if len(window_lengths) != F:
        raise ValueError("Window lengths must match the number of features in the signal.")

    if np.any(np.array(window_lengths) > S):
        raise ValueError("All filter window sizes must be less than the length of the signal.")

    # Initialize the spikes array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = [get_window(window_type, w) for w in window_lengths]
    # Copy the signal for modification
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    for feature in range(F):
        # Iterate over the signal to detect spikes
        for t in range(len(signal[:, feature])):
            # Determine the end index for the current window
            end_index = min(t + window_lengths[feature], S)

            # Extract the relevant segment of the signal and the corresponding filter window
            signal_segment = signal_copy[t:end_index, feature]
            filter_segment = filter_window[feature][: end_index - t]

            # Calculate the error for this segment
            error = np.sum(np.maximum(filter_segment - signal_segment, 0))

            # If the cumulative error is within the threshold, a spike is detected
            if error <= thresholds[feature]:
                signal_copy[t:end_index, feature] -= filter_segment
                spikes[t, feature] = 1

    if F == 1:
        spikes = spikes.flatten()
    return spikes
