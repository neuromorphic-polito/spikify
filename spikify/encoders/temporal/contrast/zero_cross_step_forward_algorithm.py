"""
.. raw:: html

    <h2>Zero Crossing Step Forward Algorithm</h2>
"""

import numpy as np


def zero_cross_step_forward(signal: np.ndarray, threshold: float | int | list[float | int] | np.ndarray) -> np.ndarray:
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
        from spikify.encoders.temporal.contrast import zero_cross_step_forward
        signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        threshold = 0.4
        encoded_signal = zero_cross_step_forward(signal, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoders.temporal.contrast import zero_cross_step_forward
        >>> signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        >>> threshold = 0.4
        >>> encoded_signal = zero_cross_step_forward(signal, threshold)
        >>> encoded_signal
        array([0, 0, 1, 0, 1, 0], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param threshold: Threshold(s) for spike generation; scalar or 1D sequence matching features.
    :type threshold: float | int | list[float | int] | numpy.ndarray
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the threshold dimensions do not match the signal features.
    :raises TypeError: If the threshold parameter is of invalid dimension.

    """

    # Check for empty signal
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    S, F = signal.shape

    # Handle threshold
    if np.isscalar(threshold):
        thresholds = np.full(F, float(threshold))
    else:
        thresholds = np.asarray(threshold, dtype=float)
        if thresholds.ndim != 1:
            raise TypeError("Threshold must be a scalar or a 1D sequence of numbers.")
        if thresholds.size != F:
            raise ValueError("Threshold must match the number of features in the signal.")

    spike = np.zeros_like(signal, dtype=np.int8)

    # Zero out negative values
    signal = np.maximum(signal, 0)

    # Apply threshold condition
    spike[signal > thresholds] = 1

    # Flatten if input was 1D
    if F == 1:
        spike = spike.flatten()

    return spike
