"""
.. raw:: html

    <h2>Phase Encoding Algorithm</h2>
"""

import numpy as np


def phase(signal: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Perform Phase Encoding (PE) on the input signal.

    This function encodes the input signal by calculating the phase angles
    of the normalized signal and quantizing these angles into a binary
    spike train representation. The encoding process uses a specified number
    of bits to determine the level of quantization.

    .. note::
        - PE requires normalized input signals between 0 and 1. If the input signal contains negative
            values, they are shifted to be non-negative and then normalized.

    Refer to the :ref:`phase_encoding_algorithm_desc` for a detailed explanation of the Phase Encoding Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.global_referenced import phase
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        num_bits = 4
        encoded_signal = phase(signal, num_bits)


    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.global_referenced import phase
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        >>> num_bits = 4
        >>> encoded_signal = phase(signal, num_bits)
        >>> encoded_signal
        array([1, 1, 1, 1, 1, 0, 0, 0], dtype=uint8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param num_bits: The number of bits to use for encoding.
    :type num_bits: int
    :return: A 1D numpy array representing the phase-encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the number of bits is not appropriate for the signal length.

    """

    # Check for invalid inputs
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    if T % num_bits != 0:
        raise ValueError(f"The phase_encoding num_bits ({num_bits}) is not a factor of the signal length ({T}).")

    signal_copy = signal.copy()

    # Normalize signal if signal has negative values
    shift = signal_copy.min(axis=0)
    shift[shift > 0] = 0  # only shift if negative values are present
    signal_copy -= shift

    # Compute max amplitude per feature to be used for scaling if max amplitude is grater than 1
    max_amp = signal_copy.max(axis=0)

    # Find features that require scaling
    features_to_scale = np.where(max_amp > 1)[0]

    for f in features_to_scale:
        signal_copy[:, f] /= max_amp[f]

    # Compute mean over the signal reshaped to bit-sized chunks
    interval_bit_mean = np.mean(signal_copy.reshape(T // num_bits, num_bits, F), axis=1)

    max_amp = interval_bit_mean.max(axis=0)

    # Find features that require scaling
    features_to_scale = np.where(max_amp > 0)[0]

    for f in features_to_scale:
        interval_bit_mean[:, f] /= max_amp[f]

    phase = np.arcsin(interval_bit_mean)

    bins = np.linspace(0, np.pi / 2, 2**num_bits + 1)
    levels = np.searchsorted(bins, phase)

    # Adjust levels to avoid out-of-range values
    levels = np.clip(levels, 0, 2**num_bits - 1)

    spikes = np.zeros((T, F), dtype=np.uint8)

    # Shift and extract bits
    # Each integer is represented in binary using `num_bits` bits.
    # The signal (levels) is right-shifted bit-by-bit to bring each bit position to the least significant bit,
    # then masked with &1 to extract it (1 if set, 0 otherwise).
    bits_arr = ((levels[..., None] >> np.arange(num_bits - 1, -1, -1)) & 1).astype(np.uint8)
    spikes = bits_arr.transpose(0, 2, 1).reshape(T, F)

    # Flatten if input was 1D
    if F == 1:
        spikes = spikes.flatten()

    return spikes
