"""
.. raw:: html

    <h2>Bens Spiker Algorithm</h2>
"""

import numpy as np
from scipy.signal import firwin
from .utils import WindowType


def bens_spiker(
    signal: np.ndarray,
    window_length: int,
    cutoff: float | np.ndarray,
    threshold: float | int | list[float, int] | np.ndarray,
    width: int | None = None,
    window_type: WindowType = "hann",
    pass_zero: bool | str = True,
    scale: bool = True,
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Ben's Spiker (BSA) encoding on the input signal.

    BSA is an efficient algorithm for converting an analog (positive-valued) signal into a unipolar spike train
    from which the original signal can be well reconstructed via convolution with the same FIR filter used during
    encoding. The method iteratively detects spike locations by comparing two cumulative error metrics over a
    sliding segment of length equal to the FIR filter:

    - error1: sum of absolute differences between the current signal segment and the filter coefficients
    - error2: sum of absolute values of the current signal segment

    A positive spike (+1) is emitted at time t if error1 ≤ error2 - threshold. When a spike is detected, the filter
    is subtracted from the corresponding signal segment, removing the detected pattern for subsequent iterations.

    Refer to the :ref:`bens_spiker_algorithm_desc` for a detailed explanation of the Ben's Spiker algorithm.

    .. note::
        - BSA requires non-negative input signals. The function automatically shifts the signal by its minimum value
          (per feature) if negative values are present.
        - The FIR filter is designed using `scipy.signal.firwin` with the specified cutoff, window type, etc.
        - When the maximum amplitute of the signal is greater than 1, filter coefficients are automatically scaled up
          (sum ≈ 2 × max amplitude) for improved dynamic range and reduced saturation (recommended practice).
        - For multi-feature signals, the same filter shape is applied across all features, but scaling is performed
          independently per feature based on its amplitude.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.deconvolution import bens_spiker
        signal = np.array([0.1, 0.2, 0.8, 0.95, 0.5, 0.3, 0.1])
        window_length = 3
        threshold = 0.1
        cutoff = 0.1
        encoded_signal, shift, fir_coeffs = bens_spiker(signal, window_length, cutoff, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.deconvolution import bens_spiker
        >>> signal = np.array([0.1, 0.2, 0.8, 0.95, 0.5, 0.3, 0.1])
        >>> window_length = 3
        >>> threshold = 0.1
        >>> cutoff = 0.1
        >>> encoded_signal, shift, fir_coeffs = bens_spiker(signal, window_length, cutoff, threshold)
        >>> encoded_signal
        array([0, 1, 1, 0, 0, 0, 0], dtype=int8)

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
    :return: Tuple containing:
             - spikes: Spike train (same shape as input, dtype=int8, values 0 or 1)
             - shift: Per-feature shift values subtracted to make signal non-negative (1D array)
             - fir_bank: Final filter coefficients used (after scaling), shape (window_length, n_features)
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the input signal is empty or if the threshold dimensions do not match the signal
                        features or if the window_lenght is greater than the signal lenght.
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
    print(features_to_scale)

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
