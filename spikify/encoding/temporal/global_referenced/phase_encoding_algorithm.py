"""
.. raw:: html

    <h2>Phase Encoding Algorithm</h2>
"""

import numpy as np


def phase_encoding(signal: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Perform phase encoding on the input signal based on the given settings.

    This function encodes the input signal by calculating the phase angles
    of the normalized signal and quantizing these angles into a binary
    spike train representation. The encoding process uses a specified number
    of bits to determine the level of quantization.

    Refer to the :ref:`phase_encoding_algorithm_desc` for a detailed explanation of the Phase Encoding Algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.global_referenced import phase_encoding
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        num_bits = 4
        encoded_signal = phase_encoding(signal, num_bits)


    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.global_referenced import phase_encoding
        >>> signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1, 0.2])
        >>> num_bits = 4
        >>> encoded_signal = phase_encoding(signal, num_bits)
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

    if len(signal) % num_bits != 0:
        raise ValueError(
            f"The phase_encoding num_bits ({num_bits}) is not a multiple of the signal length ({len(signal)})."
        )

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    S, F = signal.shape
    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to bit-sized chunks
    signal = np.mean(signal.reshape(signal.shape[0] // num_bits, num_bits, signal.shape[1]), axis=1)
    signal_max = signal.max(axis=0)  # shape (F,)

    for i in range(signal.shape[1]):  # per ogni feature
        if signal_max[i] > 0:
            signal[:, i] /= signal_max[i]

    # Compute the phase angles based on the signal
    phase = np.arcsin(signal)

    # Create phase bins and quantize the phase
    bins = np.linspace(0, np.pi / 2, 2**num_bits + 1)
    levels = np.searchsorted(bins, phase)

    # Adjust levels to avoid out-of-range values
    levels = np.clip(levels, 0, 2**num_bits - 1)

    N, F = levels.shape
    spikes = np.zeros((N * num_bits, F), dtype=np.uint8)

    # Vectorized bit extraction for all features and samples
    bits_arr = ((levels[..., None] >> np.arange(num_bits - 1, -1, -1)) & 1).astype(np.uint8)
    spikes = bits_arr.transpose(0, 2, 1).reshape(N * num_bits, F)

    if F == 1:
        spikes = spikes.flatten()

    return spikes
