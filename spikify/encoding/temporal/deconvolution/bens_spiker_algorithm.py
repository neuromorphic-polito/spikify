"""
.. raw:: html

    <h2>Bens Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal.windows import get_window
from typing import Literal

WindowType = Literal[
    "barthann",
    "brthan",
    "bth",
    "bartlett",
    "bart",
    "brt",
    "blackman",
    "black",
    "blk",
    "blackmanharris",
    "blackharr",
    "bkh",
    "bohman",
    "bman",
    "bmn",
    "boxcar",
    "box",
    "ones",
    "rect",
    "rectangular",
    "cosine",
    "halfcosine",
    "exponential",
    "poisson",
    "flattop",
    "flat",
    "flt",
    "hamming",
    "hamm",
    "ham",
    "hann",
    "han",
    "lanczos",
    "sinc",
    "nuttall",
    "nutl",
    "nut",
    "parzen",
    "parz",
    "par",
    "taylor",
    "taylorwin",
    "triangle",
    "triang",
    "tri",
    "tukey",
    "tuk",
]


def bens_spiker(
    signal: np.ndarray,
    window_lengths: int | list[int],
    threshold: float | list[float],
    window_type: WindowType = "boxcar",
) -> np.ndarray:
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

    if isinstance(window_lengths, int):
        window_lengths = [window_lengths] * F

    if len(window_lengths) != F:
        raise ValueError("Window lengths must match the number of features in the signal.")

    if np.any(np.array(window_lengths) > S):
        raise ValueError("All filter window sizes must be less than the length of the signal.")

    if isinstance(threshold, float):
        threshold = [threshold] * F

    if len(threshold) != F:
        raise ValueError("Threshold must match the number of features in the signal.")

    # Initialize the spike array
    spikes = np.zeros_like(signal, dtype=np.int8)

    # Create the boxcar filter window
    filter_window = [get_window(window_type, w) for w in window_lengths]

    # Copy of the signal to avoid modifying the original input
    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Iterate over the signal to detect spikes
    for feature in range(F):
        for t in range(len(signal[:, feature]) - window_lengths[feature] + 1):
            # Calculate errors using the filter window
            segment = signal_copy[t : t + window_lengths[feature], feature]
            error1 = np.sum(np.abs(segment - filter_window[feature]), axis=0)
            error2 = np.sum(np.abs(segment), axis=0)

            # Update signal and spike array if a spike is detected
            if error1 <= (error2 - threshold[feature]):
                signal_copy[t : t + window_lengths[feature], feature] -= filter_window[feature]
                spikes[t, feature] = 1
    if F == 1:
        spikes = spikes.flatten()

    return spikes
