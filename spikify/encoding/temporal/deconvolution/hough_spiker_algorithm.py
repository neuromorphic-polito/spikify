"""
.. raw:: html

    <h2>Hough Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal import firwin
from utils import WindowType


def hough_spiker(
    signal: np.ndarray,
    window_length: int,
    cutoff: float | np.ndarray,
    width: int | None = None,
    window_type: WindowType = "hann",
    pass_zero: bool | str = True,
    scale: bool = True,
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Hough Spiker Algorithm (HSA) encoding on the input signal.

    HSA is a simple, parameter-free technique for converting non-negative analog signals (scaled to [0, 1]) into
    unipolar spike trains that can be accurately reconstructed via convolution with the same FIR filter used during
    encoding. It iteratively scans the signal with a sliding window of length equal to the FIR filter and emits a
    positive spike (+1) at time t if the entire signal segment [t : t + window_length] is greater than or equal to
    the filter coefficients pointwise. Upon detection, the filter is subtracted from the segment, effectively removing
    the detected pattern for subsequent scans.

    .. note::
        - HSA requires non-negative inputs; automatic shifting and normalization to [0, 1] is applied per feature.
        - The FIR filter is designed using scipy.signal.firwin with the specified cutoff, window type, etc.
        - For multi-feature signals, the same filter shape is applied across all features, but scaling is performed
          independently per feature based on its amplitude.

    Refer to the :ref:`hough_spiker_algorithm_desc` for a detailed explanation of the HSA.

    **Code Example**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.deconvolution import hough_spiker
        signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1])
        window_length = 3
        cutoff = 0.1
        encoded_signal, shift, norm, fir_coeffs = hough_spiker(signal, window_length, cutoff)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import hough_spiker
        >>> signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1])
        >>> window_length = 3
        >>> cutoff = 0.1
        >>> encoded_signal, shift, norm, fir_coeffs = hough_spiker(signal, window_length, cutoff)
        >>> encoded_signal
        array([0, 1, 0, 0, 0, 0, 0], dtype=int8)

    :param signal: Input signal to encode (1D or 2D: time × features).
    :type signal: numpy.ndarray
    :param window_length: Length of the FIR filter (number of coefficients).
    :type window_length: int
    :param cutoff: Cutoff frequency(ies) for the FIR filter design (normalized 0 to 1, where 1 = Nyquist).
                   Scalar or an array of cutoff frequencies (that is, band edges).
    :type cutoff: float | numpy.ndarray
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
    :raises ValueError: If the input signal is empty or if the window_lenght is greater than the signal lenght.

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

    spikes = np.zeros_like(signal, dtype=np.int8)

    # Generate filter coefficient values according to their window length for each feature
    fir = firwin(window_length, cutoff, width=width, window=window_type, pass_zero=pass_zero, scale=scale, fs=fs)

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
        # Iterate over the signal to detect spikes
        for t in range(0, T - window_length + 1):
            # Count how many values match or exceed the filter window values
            match_count = np.sum(signal_copy[t : t + window_length, f] >= fir)

            # If all values match or exceed, a spike is detected
            if match_count == window_length:
                signal_copy[t : t + window_length, f] -= fir_bank[:, f]
                spikes[t, f] = 1

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes, shift, norm, fir_bank
