"""
.. raw:: html

    <h2>Modified Hough Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal import firwin
from .utils import WindowType


def modified_hough_spiker(
    signal: np.ndarray,
    window_length: int,
    cutoff: float | np.ndarray,
    threshold: float | int | list[float, int] | np.ndarray,
    width: int | None = None,
    window_type: WindowType = "hann",
    pass_zero: bool | str = True,
    scale: bool = True,
    fs: float | None = None,
) -> np.ndarray:
    """
    Perform Modified Hough Spiker Algorithm (MHSA) encoding on the input signal.

    The Modified Hough Spiker Algorithm (MHSA) is basically an improved version of the original Hough Spiker
    Algorithm (HSA). The original HSA is very strict: it only allows a spike to be created if the current piece of
    the signal is exactly as big as or bigger than the filter pattern at every single point in that window. If even
    one tiny spot dips below the filter, no spike is detected.

    MHSA relaxes this rule a little bit to make it more practical and flexible. Instead of demanding perfection
    everywhere, it allows a small amount of "shortfall" — places where the signal is slightly below the filter.
    It measures how much the signal falls short in those spots (by adding up the differences where the filter
    is higher than the signal), and if that total shortfall is small enough (below a chosen limit called the
    threshold), it still decides to create a spike there. Then, just like in the original, it subtracts the filter
    pattern from the signal to remove the detected spike pattern so the algorithm can keep looking for the next one.

    .. note::
        - MHSA requires non-negative inputs; automatic shifting and normalization to [0, 1] is applied per feature.
        - The FIR filter is designed using `scipy.signal.firwin` with the specified cutoff, window type, etc.
        - For multi-feature signals, the same filter shape is applied across all features, but scaling is performed
          independently per feature based on its amplitude.

    Refer to the :ref:`modified_hough_spiker_algorithm_desc` for a detailed explanation of the Modified Hough Spiker
    Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.deconvolution import modified_hough_spiker
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        window_length = 3
        threshold = 0.5
        cutoff = 0.1
        encoded_signal, shift, norm, fir_coeffs = modified_hough_spiker(signal, window_length, cutoff, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import modified_hough_spiker
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        >>> window_length = 3
        >>> threshold = 0.5
        >>> cutoff = 0.1
        >>> encoded_signal, shift, norm, fir_coeffs = modified_hough_spiker(signal, window_length, cutoff, threshold)
        >>> encoded_signal
        array([0, 0, 1, 1, 0, 0, 1], dtype=int8)

    :param signal: Input signal to encode (1D or 2D: time × features).
    :type signal: numpy.ndarray
    :param window_length: Length of the FIR filter (number of coefficients).
    :type window_length: int
    :param cutoff: Cutoff frequency(ies) for the FIR filter design (normalized 0 to 1, where 1 = Nyquist).
                   Scalar or an array of cutoff frequencies (that is, band edges).
    :type cutoff: float | numpy.ndarray
    :param threshold: Threshold factor for spike detection.
                      Scalar or per-feature sequence.
    :type threshold: float | int | list[float | int] | numpy.ndarray
    :param width: Transition width for FIR filter design (optional, used with certain window types).
    :type width: int | None
    :param window_type: Window function for FIR filter design (e.g., 'hann', 'hamming', 'blackman', 'boxcar').
    :type window_type: str
    :param pass_zero: Whether the filter should be low-pass (True) or high-pass (False/'highpass').
    :type pass_zero: bool | str
    :param scale: Set to True to scale the coefficients so that the frequency response is exactly unity at a
                  certain frequency.
    :type scale: bool
    :param fs: Sampling frequency (used for physical frequency units in cutoff; optional).
    :type fs: float | None
    :return:
        - spikes: Spike train (same shape as input, dtype=int8, values 0 or 1)
        - shift: Per-feature shift values subtracted to make signal non-negative (1D array)
        - norm: Per-feature normalization values used to scale signal to [0, 1] (1D array)
        - fir_bank: Final filter coefficients used, shape (window_length, n_features)
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the input signal is empty or if the threshold dimensions do not match the signal
                        features or if the window_length is greater than the signal lenght.
    :raises TypeError: If the threshold parameter is of invalid dimension.

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
    fir = firwin(window_length, cutoff, width=width, window=window_type, pass_zero=pass_zero, scale=scale, fs=fs)

    # Stack the same filter for all features in case we need to modify coeffiecient due to signal
    # amplitude for certain features
    fir_bank = np.stack([fir] * F, axis=0).T

    signal_copy = np.copy(np.array(signal, dtype=np.float64))

    # Normalize signal if signal has negative values
    shift = signal_copy.min(axis=0)
    shift[shift > 0] = 0  # only shift if negative values are present
    signal_copy -= shift

    norm = signal_copy.max(axis=0)
    norm[norm <= 1] = 1  # only normalize if max is greater than 1
    signal_copy /= norm

    for f in range(F):
        for t in range(0, T):

            # Determine the end index for the current window
            end_index = min(t + window_length, T)

            # Extract the relevant segment of the signal and the corresponding filter window
            signal_segment = signal_copy[t:end_index, f]
            filter_segment = fir_bank[: end_index - t, f]

            # Calculate the error for this segment
            error = np.sum(np.maximum(filter_segment - signal_segment, 0))

            # If the cumulative error is within the threshold, a spike is detected
            if error <= thresholds[f]:
                signal_copy[t:end_index, f] -= filter_segment
                spikes[t, f] = 1

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes, shift, norm, fir_bank
