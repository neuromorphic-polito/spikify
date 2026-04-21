"""
.. raw:: html

    <h2>Contrast Decoder</h2>
"""

import numpy as np


def contrast_decoder(
    spikes: np.ndarray,
    thresholds: np.ndarray,
    start_point: float | int | list[float | int] | np.ndarray,
) -> np.ndarray:
    """
    Perform Contrast family decoding on the input spike train.

    This function takes a spike train encoded with any of the Contrast family algorithms,
    Threshold-Based Representation (TBR), Step-Forward (SF), Moving Window (MW) or Zero-Crossing Step-Forward (ZCSF),
    and reconstructs the original continuous signal. The reconstruction is performed by starting from the initial
    signal value and applying the threshold incrementally based on the spike values.

    .. note::
        - The ``thresholds`` parameter should be the ``thresholds`` array returned directly by the encoder
          (TBR, SF, or MW). All three encoders return this array as part of their output to ensure the step
          size used for reconstruction exactly matches the one used during encoding.
        - The ``start_point`` parameter should be set to the first sample of the original signal prior to
          encoding (e.g. ``signal[0]`` for a 1D signal, or ``signal[0, :]`` for a multi-feature/channel signal).
          Since the Contrast family encodes only differences between consecutive samples, the absolute signal
          level is lost during encoding and must be restored by anchoring reconstruction to this initial value.


    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoders.temporal.contrast import moving_window
        from spikify.decoders.temporal.contrast import contrast_decoder

        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        window_length = 3
        spikes, thresholds = moving_window(signal, window_length, threshold=0.2)
        reconstructed_signal = contrast_decoder(spikes, thresholds, start_point=signal[0])

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoders.temporal.contrast import moving_window
        >>> from spikify.decoders.temporal.contrast import contrast_decoder
        >>> signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        >>> spikes, thresholds = moving_window(signal, window_length=3, threshold=0.2)
        >>> reconstructed_signal = contrast_decoder(spikes, thresholds, start_point=signal[0])
        >>> reconstructed_signal.flatten()
        array([0.1, 0.1, 0.1, 0.3, 0.5, 0.7])

    :param spikes: The input spike train to be decoded. This should be a numpy ndarray with values in {-1, 0, +1},
        as produced by any of the Contrast family encoders (TBR, SF, MW, ZCSF).
    :type spikes: numpy.ndarray
    :param threshold: Per-feature or channels threshold values used during encoding, as returned directly by the TBR,
        SF, MW or ZCSF encoder. Passing the encoder's returned ``thresholds`` array
        ensures the reconstruction step size exactly matches the one used during encoding.
    :type threshold: numpy.ndarray
    :param start_point: Initial signal value(s) for reconstruction (scalar or 1D sequence matching features).
        Should be set to the first sample of the original signal before encoding (e.g. ``signal[0]``), as the
        Contrast family encodes only signal differences and the absolute offset must be restored manually.
    :type start_point: float | int | list[float | int] | numpy.ndarray
    :return: A numpy array representing the reconstructed continuous signal.
    :rtype: numpy.ndarray
    :raises ValueError: If the input spike train is empty or if the start_point dimensions do not match
        the spike train feature dimensions.

    """
    # Check for empty spike train
    if len(spikes) == 0:
        raise ValueError("Spike train cannot be empty.")

    # Ensure 2D processing (T, F)
    if spikes.ndim == 1:
        spikes = spikes.reshape(-1, 1)

    T, F = spikes.shape

    # Handle startpoint
    if np.isscalar(start_point):
        start_points = np.full(F, float(start_point))
    else:
        start_points = np.asarray(start_point, dtype=float)
        if start_points.ndim != 1:
            raise ValueError("Startpoint must be a scalar or a 1D sequence of numbers.")
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

    return signal
