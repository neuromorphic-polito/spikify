"""
.. raw:: html

    <h2>Poisson Algorithm</h2>
"""

import numpy as np


def poisson(signal: np.ndarray, interval_length: int, seed: int = 0) -> np.ndarray:
    """
    Perform Poisson encoding on the input signal.

    This function generates a spike train using a Poisson distribution, where the probability of emitting
    a spike in a given interval is determined by the normalized rate of the signal.

    See the :ref:`poisson_algorithm_desc` description for a detailed explanation of the Poisson encoding
    algorithm.

    .. note::
        - If ``interval_length == 1``, each input value is treated directly as the firing rate
          (after normalization/clipping), resulting in a standard bin-by-bin
          approximation of Poisson encoding (at most one spike per time step).
        - If ``interval_length > 1``, the average value over each interval block determines
          the constant firing rate for that block. Spikes are placed within the block using
          exponentially distributed inter-spike intervals, allowing multiple spikes per block
          (exact Poisson generation for the block's constant rate).

    **Code Example:**

    .. code-block:: python

            import numpy as np
            from spikify.encoding.rate import poisson
            signal = np.array([0.2, 0.5, 0.8, 1.0])
            np.random.seed(0)
            interval_length = 2
            encoded_signal = poisson(signal, interval_length)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.rate import poisson
        >>> # Example with numpy array
        >>> signal = np.array([0.2, 0.5, 0.8, 1.0])
        >>> interval_length = 2
        >>> encoded_signal = poisson(signal, interval_length)
        >>> encoded_signal
        array([0, 0, 0, 1], dtype=int8)

    :param signal: Input signal to encode (1D or 2D: timestamps × features).
    :type signal: numpy.ndarray
    :param interval_length: Length of each time block for rate computation and spike placement.
                            Must evenly divide the signal length. ``interval_length=1`` uses the
                            instantaneous value as rate (bin-by-bin); larger values use the block
                            mean as constant rate (allows multiple spikes per block).
    :type interval_length: int
    :param seed: Random seed for reproducibility. Default is 0.
    :type seed: int
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if signal length is not divisible for the interval length
    :raises TypeError: If the signal is not a numpy.ndarray

    """

    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    if T % interval_length != 0:
        raise ValueError(
            f"The interval ({interval_length}) must evenly divide the signal length ({T}). "
            "Consider trimming or padding the signal to make its length a multiple of the interval."
        )

    # Set seed
    np.random.seed(seed)

    signal_copy = np.copy(signal)

    # Ensure non-negative signal values
    signal_copy = np.clip(signal_copy, 0, None)

    max_amp = signal_copy.max(axis=0)

    # Find features that require scaling
    features_to_scale = np.where(max_amp > 1)[0]

    for f in features_to_scale:
        signal_copy[:, f] /= max_amp[f]

    # Compute mean over the signal reshaped to interval-sized chunks
    interval_rate_mean = np.mean(signal_copy.reshape(T // interval_length, interval_length, F), axis=1)

    spikes = np.zeros((T // interval_length, interval_length, F), dtype=np.int8)

    # Create bins for Poisson encoding
    bins = np.linspace(0, 1, interval_length + 1)

    for feat in range(F):
        for idx, rate in enumerate(interval_rate_mean[:, feat]):
            if rate > 0:
                ISI = -np.log(1 - np.random.random(interval_length)) / (  # inter-spike intervals where probability of
                    rate * interval_length  # having k=0 spikes is equal to rate (time
                )  # amount to wait to see the next spike)
                spike_times = np.searchsorted(bins, np.cumsum(ISI)) - 1  # find spike times
                spike_times = spike_times[spike_times < interval_length]  # clip times within interval
                spikes[idx, spike_times, feat] = 1

    spikes = spikes.reshape(T, F)

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes
