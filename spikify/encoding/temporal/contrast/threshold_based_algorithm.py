"""
.. raw:: html

    <h2>Threshold Based Representation Algorithm</h2>
"""

import numpy as np
from typing import Tuple


def threshold_based_representation(
    signal: np.ndarray, factor: float | int | list[float | int]
) -> Tuple[np.ndarray, np.ndarray]:
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
        array([ 1,  0,  -1,  1,  0,  0], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param factor: The factor value (`factor`) that controls the noise-reduction threshold.
                   Can be a float, an integer, or a list of floats or integers.
    :type factor: float | int | list[float | int]
    :return: A tuple containing the encoded spike train and the computed threshold for each feature.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the input signal is empty.
    :raises TypeError: If the signal is not a numpy ndarray.

    """

    # Input validation
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    # Ensure 2D processing (T, F)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    T, F = signal.shape

    # Handle factor
    if isinstance(factor, (int, float)):
        factors = np.full(F, float(factor))
    elif isinstance(factor, list):
        if len(factor) != F:
            raise ValueError("Factor list length must match the number of features in the signal.")
        if not all(isinstance(f, (int, float)) for f in factor):
            raise TypeError("All elements in factor list must be numeric.")
        factors = np.array(factor, dtype=float)
    else:
        raise TypeError("factor must be a float or a list of floats.")

    spike = np.zeros_like(signal, dtype=np.int8)

    # Compute variation exactly as in the original code
    # diff[t] = s[t+1] - s[t] for t = 0 to T-2
    # diff[T-1] = diff[T-2] (last value set to second-last)
    diff = np.diff(signal, axis=0, append=signal[[0], :])  # append first value of signal to maintain shape
    diff[-1, :] = diff[-2, :]  # force last to equal second-last

    print(diff.shape)

    # Compute threshold per feature (over all T variations, including the duplicated last)
    threshold = np.mean(diff, axis=0) + factors * np.std(diff, axis=0)

    print(threshold.shape)

    # Generate spikes: compare on the full diff array (length S)
    threshold = threshold.reshape(1, threshold.shape[0])
    spike[diff > threshold] = 1
    spike[diff < -threshold] = -1

    # Flatten if input was 1D
    if F == 1:
        spike = spike.flatten()

    return spike, threshold.flatten()
