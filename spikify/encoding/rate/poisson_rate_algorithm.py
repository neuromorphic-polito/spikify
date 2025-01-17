"""
.. raw:: html

    <h2>Poisson Rate Algorithm</h2>
"""

import numpy as np


def poisson_rate(signal: np.ndarray, interval_length: int) -> np.ndarray:
    """
    Perform Poisson rate encoding on the input signal.

    This function generates a spike train using a Poisson distribution,
    where the probability of emitting a spike in a given interval is determined by the normalized rate of the signal.

    See the :ref:`poisson_rate_algorithm_desc` description for a detailed explanation of the Poisson rate encoding
    algorithm.

    **Code Example:**

    .. code-block:: python

            import numpy as np
            from spikify.encoding.rate import poisson_rate
            signal = np.array([0.2, 0.5, 0.8, 1.0])
            np.random.seed(0)
            interval_length = 2
            encoded_signal = poisson_rate(signal, interval_length)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoding.rate import poisson_rate
        >>> # Example with numpy array
        >>> signal = np.array([0.2, 0.5, 0.8, 1.0])
        >>> # Set seed for reproducibility (optional)
        >>> np.random.seed(0)
        >>> interval_length = 2
        >>> encoded_signal = poisson_rate(signal, interval_length)
        >>> encoded_signal
        array([0, 0, 1, 1], dtype=int8)

    :param signal: The input signal to be encoded. This should be a numpy ndarray.
    :type signal: numpy.ndarray
    :param interval_length: The size of the interval for encoding the spike train.
    :type interval_length: int
    :return: A 1D array or tensor of encoded spike data after Poisson rate encoding.
    :rtype: numpy.ndarray
    :raises ValueError: If the input signal is empty.
    :raises ValueError: If the interval is not a factor of the signal length.
    :raises TypeError: If the signal is not a numpy.ndarray

    """
    # Check for invalid inputs
    if signal.shape[0] == 0:
        raise ValueError("Signal cannot be empty.")

    if signal.shape[0] % interval_length != 0:
        raise ValueError(
            f"The interval ({interval_length}) is not a factor of the signal length ({signal.shape[0]}). "
            "To resolve this, consider trimming or padding the signal to ensure its length is a multiple of the "
            "interval."
        )

    np.random.seed(0)

    # Ensure non-negative signal values
    signal = np.clip(signal, 0, None)

    # Compute mean over the signal reshaped to interval-sized chunks
    signal = np.mean(signal.reshape(-1, interval_length), axis=1)

    # Normalize the signal
    signal_max = signal.max()
    if signal_max > 0:
        signal /= signal_max

    # Initialize the spike array
    spikes = np.zeros((signal.shape[0], interval_length), dtype=np.int8)

    # Create bins for Poisson rate encoding
    bins = np.linspace(0, 1, interval_length + 1)

    # Generate Poisson spike trains
    for i, rate in enumerate(signal):
        if rate > 0:
            ISI = [
                -np.log(1 - np.random.random()) / (rate * interval_length) for _ in range(interval_length)
            ]  # Inter-spike intervals
            spike_times = np.searchsorted(bins, np.cumsum(ISI)) - 1  # Find spike times
            spike_times = spike_times[spike_times < interval_length]  # Clip times within interval
            spikes[i, spike_times] = 1

    # Flatten the 2D array into a 1D spike train
    spikes = spikes.flatten()
    return spikes
