"""
.. raw:: html

    <h2>Bens Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal import firwin
from .utils import WindowType

# TODO: Add new parameters to the function to allow for more customization of the FIR filter
#       (e.g., cutoff frequency, fs, etc.). Additionaly add a a new parameter to scale the filter
#       to twice the max amplitude of the signal (as done in Petro et al.). This has to be done when
#       the signal in input is non-negative and not in the range [0, 1]. If the signal is already positive
#       and in the range the filter coefficient scaling should be skipped.


def bens_spiker(
    signal: np.ndarray,
    window_length: int | list[int] | np.ndarray,
    threshold: float | int | list[float, int] | np.ndarray,
    window_type: WindowType = "boxcar",
    scale_coeffs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
        signal = np.array([0.1, 0.2, 2.0, 1.0, 0.5, 0.3, 0.1])
        window_length = 3
        threshold = 0.5
        spikes = bens_spiker(signal, window_length, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import bens_spiker
        >>> signal = np.array([0.1, 0.2, 2.0, 1.0, 0.5, 0.3, 0.1])
        >>> window_length = 3
        >>> threshold = 0.5
        >>> spikes = bens_spiker(signal, window_length, threshold)
        >>> spikes
        array([0, 1, 0, 0, 0, 0, 0], dtype=int8)

    :param signal: The input signal to be analyzed. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param window_length: Integer scalar (broadcast to all features) or 1‑D sequence/ndarray for filter window.
    :type window_length: int | list[int] | numpy.ndarray
    :param threshold: Threshold(s) for spike generation; scalar or 1D sequence matching features.
    :type threshold: float | int | list[float | int] | numpy.ndarray
    :return: A tuple containing a numpy array representing the detected spikes and the shift used for normalization.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the input signal is empty or if any window length exceeds the signal length.
    :raises TypeError: If window_length or threshold are not of the expected types.

    """

    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    # Handle window_length
    if np.isscalar(window_length):
        window_lengths = np.full(F, int(window_length), dtype=int)
    else:
        window_lengths = np.asarray(window_length)
        if window_lengths.ndim != 1:
            raise TypeError("window_length must be a scalar or a 1D sequence of integers.")
        if window_lengths.size != F:
            raise ValueError("window_lengths must match the number of features in the signal.")

    if np.any(window_lengths > T):
        raise ValueError("All filter window sizes must be less than the number of time steps in the signal.")

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

    # Generate filter values according to their window length for each feature
    filter = [firwin(numtaps=w, cutoff=0.2, window=window_type, fs=44100, scale=True) for w in window_lengths]

    signal_copy = np.copy(signal)

    # Normalize signal
    shift = signal_copy.min(axis=0)
    signal_copy -= shift

    max_amp = 2 * signal_copy.max(axis=0)

    # Scale filter coefficients if required
    if scale_coeffs:
        for f in range(F):
            s = filter[f].sum()
            filter[f] *= max_amp[f] / s

    for f in range(F):
        for t in range(0, T - window_lengths[f] + 1):
            seg = signal_copy[t : t + window_lengths[f], f]  # segment of the signal
            error1 = np.abs(seg - filter[f]).sum()  # error between segment and filter
            error2 = np.abs(seg).sum()  # error between segment and zero signal
            if error1 <= (error2 - thresholds[f]):  # spike condition
                spikes[t, f] = 1
                signal_copy[t : t + window_lengths[f], f] -= filter[f]  # update signal by removing filter effect

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes, shift, max_amp
