"""
.. raw:: html

    <h2>Step Forward Algorithm</h2>
"""

import numpy as np


def step_forward(signal: np.ndarray, threshold: float | list[float]) -> np.ndarray:
    """
    Perform Step-Forward encoding on the input signal.

    This function takes a continuous signal and converts it into a spike train using a dynamically updated baseline
    and threshold-based approach. A spike is generated when the signal exceeds or drops below the dynamically
    adjusted baseline (`Base`) by the specified `Threshold`.

    Refer to the :ref:`step_forward_algorithm_desc` for a detailed explanation of the Step-Forward encoding algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.contrast import step_forward
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        threshold = 0.2
        encoded_signal = step_forward(signal, threshold)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.contrast import step_forward
        >>> signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        >>> threshold = 0.2
        >>> encoded_signal = step_forward(signal, threshold)
        >>> encoded_signal
        array([0, 0, 1, 0, 0, 1], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param threshold: The threshold value(s) for spike detection. Can be a float or a list of floats.
    :type threshold: float | list[float]
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty.
    :raises TypeError: If the signal is not a numpy ndarray.

    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    S, F = signal.shape

    if isinstance(threshold, float):
        thresholds = [threshold] * F
    elif isinstance(threshold, list):
        if not all(isinstance(w, float) for w in threshold):
            raise TypeError("All elements in threshold list must be float.")
        thresholds = threshold
    else:
        raise TypeError("Threshold must be a float or a list of floats.")

    if len(thresholds) != F:
        raise ValueError("Threshold must match the number of features in the signal.")

    spike = np.zeros_like(signal, dtype=np.int8)

    # Base value initialized at the start of the signal
    for feat in range(F):
        base = signal[0, feat]
        for value_idx, value in enumerate(signal[:, feat]):
            if value > base + thresholds[feat]:
                spike[value_idx, feat] = 1
                base += thresholds[feat]
            elif value < base - thresholds[feat]:
                spike[value_idx, feat] = -1
                base -= thresholds[feat]

    if F == 1:
        spike = spike.flatten()
    return spike
