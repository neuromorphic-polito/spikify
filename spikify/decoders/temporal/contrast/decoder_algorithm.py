"""
.. raw:: html

    <h2>Contrast Decoder</h2>
"""

import numpy as np


def contrast_decoder(
    spikes: np.ndarray,
    threshold: float | int | list[float | int] | np.ndarray,
    start_point: float | int | list[float | int] | np.ndarray,
) -> np.ndarray:
    """
    Perform Contrast family decoding on the input spike train.

    This function takes a spike train encoded with any of the Contrast family algorithms,
    Threshold-Based Representation (TBR), Step-Forward (SF), or Moving Window (MW), and reconstructs the original
    continuous signal. The reconstruction is performed by starting from the initial signal value and applying the
    threshold incrementally based on the spike values.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.decoders.temporal.contrast import contrast_decoder

        spikes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        threshold = 0.2
        startpoint = 0.1
        reconstructed_signal = contrast_decoder(spikes, threshold, startpoint)
        reconstructed_signal

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.decoders.temporal.contrast import contrast_decoder
        >>> spikes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        >>> threshold = 0.2
        >>> startpoint = 0.1
        >>> reconstructed_signal = contrast_decoder(spikes, threshold, startpoint)
        >>> reconstructed_signal
        array([0.1, 0.1, 0.1, 0.3, 0.5, 0.7])

    :param spikes: The input spike train to be decoded. This should be a numpy ndarray with values in {-1, 0, +1},
        as produced by any of the Contrast family encoders (TBR, SF, MW).
    :type spikes: numpy.ndarray
    :param threshold: Threshold(s) used during encoding; scalar or 1D sequence matching features.
    :type threshold: float | int | list[float | int] | numpy.ndarray
    :param start_point: Initial signal value(s) for reconstruction; scalar or 1D sequence matching features.
    :type start_point: float | int | list[float | int] | numpy.ndarray
    :return: A numpy array representing the reconstructed continuous signal.
    :rtype: numpy.ndarray
    :raises ValueError: If the input spike train is empty or if the threshold/start_point dimensions do not match
        the spike train features.
    :raises TypeError: If the threshold or start_point parameter is of invalid dimension.

    """
    # Check for empty spike train
    if len(spikes) == 0:
        raise ValueError("Spike train cannot be empty.")

    # Ensure 2D processing (T, F)
    if spikes.ndim == 1:
        spikes = spikes.reshape(-1, 1)

    T, F = spikes.shape

    # Handle threshold
    if np.isscalar(threshold):
        thresholds = np.full(F, float(threshold))
    else:
        thresholds = np.asarray(threshold, dtype=float)
        if thresholds.ndim != 1:
            raise TypeError("Threshold must be a scalar or a 1D sequence of numbers.")
        if thresholds.size != F:
            raise ValueError("Threshold must match the number of features in the spike train.")

    # Handle startpoint
    if np.isscalar(start_point):
        start_points = np.full(F, float(start_point))
    else:
        start_points = np.asarray(start_point, dtype=float)
        if start_points.ndim != 1:
            raise TypeError("Startpoint must be a scalar or a 1D sequence of numbers.")
        if start_points.size != F:
            raise ValueError("Startpoint must match the number of features in the spike train.")

    signal = np.zeros((T, F), dtype=float)
    signal[0] = start_points

    for t in range(1, T):
        for f in range(F):
            if spikes[t, f] == 1:
                signal[t, f] = signal[t - 1, f] + thresholds[f]
            elif spikes[t, f] == -1:
                signal[t, f] = signal[t - 1, f] - thresholds[f]
            else:
                signal[t, f] = signal[t - 1, f]

    # Flatten if input was 1D
    if F == 1:
        signal = signal.flatten()

    return signal
