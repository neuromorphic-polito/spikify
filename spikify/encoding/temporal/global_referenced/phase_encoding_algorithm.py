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

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to bit-sized chunks
    signal = np.mean(signal.reshape(-1, num_bits), axis=1)

    # Normalize the signal if the maximum is greater than 0
    signal_max = signal.max()
    if signal_max > 0:
        signal /= signal_max

    # Compute the phase angles based on the signal
    phase = np.arcsin(signal)

    # Create phase bins and quantize the phase
    bins = np.linspace(0, np.pi / 2, 2**num_bits + 1)
    levels = np.searchsorted(bins, phase)

    # Adjust levels to avoid out-of-range values
    levels = np.clip(levels, 0, 2**num_bits - 1)

    # Convert levels to binary and flatten the result to a 1D spike array
    spikes = np.unpackbits(levels[:, None].astype(np.uint8), axis=1, bitorder="big")[:, -num_bits:].reshape(-1)

    return spikes
