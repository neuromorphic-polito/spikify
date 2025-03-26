"""
.. raw:: html

    <h2>Zero Crossing Step Forward Algorithm</h2>
"""

import numpy as np


def zero_cross_step_forward(signal: np.ndarray, threshold: int) -> np.ndarray:
    """
    Perform Zero-Crossing Step-Forward (ZCSF) encoding on the input signal.

    This function generates a spike train based on the positive values of the input signal that exceed a specified
    threshold. Negative values of the signal are zeroed out (half-wave rectification), and only positive spikes are
    considered.

    Refer to the :ref:`zero_cross_step_forward_algorithm_desc` for a detailed explanation of the ZCSF encoding
    algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.contrast import zero_cross_step_forward
        signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        threshold = 0.4
        encoded_signal = zero_cross_step_forward(signal, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.contrast import zero_cross_step_forward
        >>> signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        >>> threshold = 0.4
        >>> encoded_signal = zero_cross_step_forward(signal, threshold)
        >>> encoded_signal
        array([0, 0, 1, 0, 1, 0], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param threshold: The threshold value used to determine spike generation.
    :type threshold: int
    :return: A 1D numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty.
    :raises TypeError: If the signal is not a numpy ndarray.

    """

    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    spike = np.zeros_like(signal, dtype=np.int8)

    # Zero out negative values
    signal = np.maximum(signal, 0)

    # Apply threshold condition
    spike[signal > threshold] = 1

    return spike
