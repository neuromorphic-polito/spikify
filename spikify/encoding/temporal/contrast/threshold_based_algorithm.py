"""
.. raw:: html

    <h2>Threshold Based Representation Algorithm</h2>
"""

import numpy as np


def threshold_based_representation(signal: np.ndarray, factor: float) -> np.ndarray:  # multiple
    """
    Perform Threshold-Based Representation (TBR) encoding on the input signal.

    This function takes a continuous signal and converts it into a spike train using a fixed threshold based on
    the signal's variations. A spike is generated when the variation exceeds the computed threshold.

    Refer to the :ref:`threshold_based_representation_algorithm_desc` for a detailed explanation of the TBR
    encoding algorithm.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoding.temporal.contrast import threshold_based_representation
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        factor = 0.5
        encoded_signal = threshold_based_representation(signal, factor)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.temporal.contrast import threshold_based_representation
        >>> signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        >>> factor = 0.5
        >>> encoded_signal = threshold_based_representation(signal, factor)
        >>> encoded_signal
        array([ 0,  1,  0, -1,  1,  0], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param factor: The factor value (`γ`) that controls the noise-reduction threshold.
                   Can be a float or a list/array of floats.
    :type factor: float
    :return: A 1D numpy array representing the encoded spike train.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty.
    :raises TypeError: If the signal is not a numpy ndarray.

    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if isinstance(factor, float):
        factors = [factor]
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if len(factors) != signal.shape[1]:
        raise ValueError("Factor must match the number of features in the signal.")

    spike = np.zeros_like(signal, dtype=np.int8)
    variation = np.diff(signal[1:, :], prepend=signal[[0], :], axis=0)
    threshold = np.mean(variation, axis=0) + factors * np.std(variation, axis=0)
    variation = np.insert(variation, 0, variation[1, :], axis=0)

    # Apply threshold conditions
    threshold = threshold.reshape(1, threshold.shape[0])
    spike[variation > threshold] = 1
    spike[variation < -threshold] = -1
    if spike.shape[-1] == 1:
        spike = spike.flatten()
    return spike
