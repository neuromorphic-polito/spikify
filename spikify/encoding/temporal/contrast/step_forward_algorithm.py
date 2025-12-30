"""
.. raw:: html

    <h2>Step Forward Algorithm</h2>
"""

import numpy as np


def step_forward(signal: np.ndarray, threshold: float | int | list[float | int] | np.ndarray) -> np.ndarray:
    """
    Perform Step-Forward (SF) encoding on the input signal.

    This function takes a continuous signal and converts it into a spike train using a dynamically updated baseline
    signal and threshold-based approach. A spike is generated when the signal exceeds or drops below the dynamically
    adjusted baseline (`base`) by the specified `threshold`.

    Refer to the :ref:`step_forward_algorithm_desc` for a detailed explanation of the SF encoding algorithm.

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
    :type threshold: float | int | list[float | int] | numpy.ndarray
    :return: A numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty or if the threshold dimensions do not match the signal features.
    :raises TypeError: If the threshold parameter is of invalid dimension.

    """

    # Input validation
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

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

    # base signal initialized at the start of the signal
    base = signal[0, :]

    # Iterate over signal values skipping the first timestep since it's used for initialization of the base signal
    for feat in range(F):
        base = signal[0, feat]
        for t in range(1, T):
            value = signal[t, feat]
            if value > base + thresholds[feat]:
                spike[t, feat] = 1
                base += thresholds[feat]
            elif value < base - thresholds[feat]:
                spike[t, feat] = -1
                base -= thresholds[feat]

    # Flatten if input was 1D
    if F == 1:
        spike = spike.flatten()

    return spike
