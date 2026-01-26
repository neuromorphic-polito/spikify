"""
.. raw:: html

    <h2>Burst Coding Algorithm</h2>
"""

import numpy as np


def burst_coding(signal: np.ndarray, n_max: int, t_min: int, t_max: int, interval_length: int) -> np.ndarray:
    """
    Perform Burst enCoding (BC) on the input signal.

    TThis function implements a biologically inspired burst coding scheme where each input
    intensity (normalized P ∈ [0, 1]) is converted into a burst of spikes:

    - Number of spikes N_s(P) = ceil(P * n_max)
    - ISI(P): Fixed at t_max when N_s = 1; otherwise linearly mapped between t_min (strong input)
      and t_max (weak input) for N_s > 1.

    The signal is divided into non-overlapping blocks of length ``interval_length``. The mean
    intensity per block determines the burst parameters (N_s and ISI). The burst is placed at
    regular intervals starting from the beginning of the output block.

    Larger intensities produce bursts with more spikes and smaller ISIs. The ``interval_length``
    must be large enough to fit the longest burst.

    .. note::
        - BC requires normalized input signals between 0 and 1. If the input signal contains negative values, they are
            shifted to be non-negative and then normalized.

    Refer to the :ref:`burst_coding_algorithm_desc` for a detailed explanation of the Burst Coding Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.latency import burst_coding
        np.random.seed(42)
        signal = np.random.rand(16)
        n_max = 4
        t_min = 2
        t_max = 6
        length = 16
        encoded_signal = burst_coding(signal, n_max, t_min, t_max, length)


    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.latency import burst_encoding
        >>> np.random.seed(42)
        >>> signal = np.random.rand(16)
        >>> n_max = 4
        >>> t_min = 2
        >>> t_max = 6
        >>> length = 16
        >>> encoded_signal = burst_encoding(signal, n_max, t_min, t_max, length)
        >>> encoded_signal
        array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=int8)

    :param signal: Input signal to encode (1D or 2D: timestamps × features).
    :type signal: numpy.ndarray
    :param n_max: Maximum number of spikes in a burst (for P = 1). Must be ≥ 1.
    :type n_max: int
    :param t_min: Minimum inter-spike interval (ISI) in time steps (for strong inputs with N_s > 1).
    :type t_min: int
    :param t_max: Maximum inter-spike interval (ISI) in time steps (for weak inputs or N_s = 1).
    :type t_max: int
    :param interval_length: Length of each output block corresponding to one input block.
                            Must divide the signal length evenly and be large enough to fit
                            the longest burst (approximately n_max * (t_min + 1)).
    :type interval_length: int
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If signal is empty, interval_length does not divide signal length,
                        or interval_length is too small for the longest possible burst.

    """
    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    if T % interval_length != 0:
        raise ValueError(f"The interval_length ({interval_length}) is not a factor of the signal length ({T}).")

    signal_copy = signal.copy()

    # signal_copy = np.clip(signal_copy, 0, None)
    shift = signal_copy.min(axis=0)
    shift[shift > 0] = 0  # only shift if negative values are present
    signal_copy -= shift

    # Compute max amplitude per feature to be used for scaling if max amplitude is grater than 1
    max_amp = signal_copy.max(axis=0)

    # Find features that require scaling
    features_to_scale = np.where(max_amp > 1)[0]

    for f in features_to_scale:
        signal_copy[:, f] /= max_amp[f]

    signal_copy = np.mean(signal_copy.reshape(-1, interval_length, F), axis=1)

    spike_num = np.ceil(signal_copy * n_max).astype(int)
    ISI = np.ceil(t_max - signal_copy * (t_max - t_min)).astype(int)

    required_length = np.max(spike_num * (ISI + 1))
    if interval_length < required_length:
        raise ValueError(f"Invalid stream length, the min length is {required_length}")

    spikes = np.zeros((T // interval_length, interval_length, F), dtype=np.int8)

    for i in range(signal_copy.shape[0]):
        for f in range(F):
            spike_times = np.arange(0, spike_num[i, f] * (ISI[i, f] + 1), ISI[i, f] + 1)
            spikes[i, spike_times[spike_times < interval_length], f] = 1

    spikes = spikes.reshape(-1, F)

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes
