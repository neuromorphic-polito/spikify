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


# BSA allows only one FIR filter to be used for all signal features.
# To apply feature-specific window lengths or other filter parameters,
# the signal must be processed individually per feature
def bens_spiker(
    signal: np.ndarray,
    window_length: int,
    cutoff: float | np.ndarray,
    threshold: float | int | list[float, int] | np.ndarray,
    window_type: WindowType = "hann",
    pass_zero: bool | str = True,
    scale: bool = True,
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform spike detection using Bens Spiker Algorithm.

    This function detects spikes in an input signal based on the comparison of cumulative errors calculated over a
    segment of the signal, which is filtered using a boxcar window. A spike is detected if the cumulative error between
    the filtered signal and the raw signal is below a certain threshold.

    Refer to the :ref:`bens_spiker_algorithm_desc` for a detailed explanation of the Ben's Spiker algorithm.

    .. note::

        Although ``bens_spiker`` supports multi-feature signals, the same filter configuration is applied
        across all features.

        For applications requiring heterogeneous filter configurations across features, it is
        recommended to apply this algorithm feature by feature.

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
    :param window_length: Window size for the FIR filter.
    :type window_length: int
    :param window_type: Type of window to use for the FIR filter. Default is 'boxcar'.
    :type window_type: str
    :param scale_coeffs: Whether to scale the filter coefficients to match twice the maximum amplitude of the signal.
    :type scale_coeffs: bool
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
    if window_length > T:
        raise ValueError("window_length must be less than the number of time steps in the signal.")

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

    # Generate filter coefficient values according to their window length for each feature
    fir = firwin(window_length, cutoff, window=window_type, pass_zero=pass_zero, scale=scale, fs=fs)

    # Stack the same filter for all features in case we need to modify coeffiecient due to signal
    # amplitude for certain features
    fir_bank = np.stack([fir] * F, axis=0).T

    signal_copy = np.copy(signal)

    # Normalize signal if signal has negative values
    shift = signal_copy.min(axis=0)
    shift[shift > 0] = 0  # only shift if negative values are present
    signal_copy -= shift

    # Compute max amplitude per feature to be used for scaling if max amplitude is grater than 1
    max_amp = signal_copy.max(axis=0)

    # Find features that require scaling
    features_to_scale = np.where(max_amp > 1)[0]

    for f in features_to_scale:
        s = fir_bank[:, f].sum()
        fir_bank[:, f] *= 2 * max_amp[f] / s

    for f in range(F):
        for t in range(0, T - window_length + 1):
            seg = signal_copy[t : t + window_length, f]  # segment of the signal
            error1 = np.abs(seg - fir_bank[:, f]).sum()  # error between segment and filter
            error2 = np.abs(seg).sum()  # error between segment and zero signal
            if error1 <= (error2 - thresholds[f]):  # spike condition
                spikes[t, f] = 1
                signal_copy[t : t + window_length, f] -= fir_bank[:, f]  # update signal by removing filter effect

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes, shift, fir_bank
