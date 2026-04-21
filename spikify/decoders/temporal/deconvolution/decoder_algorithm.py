"""
.. raw:: html

    <h2>Deconvolution Decoder</h2>
"""

import numpy as np
from scipy.signal import lfilter


def deconvolution_decoder(
    spikes: np.ndarray,
    fir_bank: np.ndarray,
    shift: np.ndarray,
    norm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform signal reconstruction via deconvolution decoding of a spike train.

    The Deconvolution Decoder reverses the encoding performed by spike-based deconvolution algorithms such as
    the Hough Spiker (HSA), its Modified variant (MHSA) and Bens Spiker (BSA). Given a binary spike train and the FIR
    filter bank produced during encoding, it reconstructs an approximation of the original continuous signal
    by applying each feature's filter to its corresponding spike train using causal linear filtering.

    .. note::
        - The ``fir_bank``, ``shift``, and ``norm`` parameters should be the values returned directly by the
          encoder algorithm to ensure accurate inversion of all pre-processing steps.
        - If ``norm`` is ``None``, no amplitude rescaling is applied; only the per-feature shift is restored.
          Pass ``norm`` explicitly whenever the encoder performed normalization. Only MHSA and HSA have normalization
          steps, while BSA does not.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.encoders.temporal.deconvolution import modified_hough_spiker
        from spikify.decoders.temporal.deconvolution import deconvolution_decoder

        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        window_length = 3
        threshold = 0.5
        cutoff = 0.1
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length, cutoff, threshold)
        reconstructed = deconvolution_decoder(spikes, fir_bank, shift, norm)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.encoders.temporal.deconvolution import modified_hough_spiker
        >>> from spikify.decoders.temporal.deconvolution import deconvolution_decoder
        >>> signal = np.array([0.0, 0.0, 0.5, 0.9, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> window_length = 5
        >>> threshold = 0.3
        >>> cutoff = 0.1
        >>> spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length, cutoff, threshold)
        >>> reconstructed = deconvolution_decoder(spikes, fir_bank, shift, norm)
        >>> reconstructed.flatten()
        array([0.        , 0.24793707, 0.50412586, 0.49587414, 0.50412586,
               0.24793707, 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.24793707])

    :param spikes: Binary spike train to decode (values in {0, 1}), as produced by
        any of the deconvolution family encoders (HSA, MHSA, BSA).
    :type spikes: numpy.ndarray
    :param fir_bank: FIR filter coefficients used during encoding.
        Each column corresponds to the filter applied to one feature or channel. This is the ``fir_bank`` value
        returned directly by the encoder.
    :type fir_bank: numpy.ndarray
    :param shift: Per-feature shift values subtracted during encoding to ensure signal non-negativity.
                  This is the ``shift`` value returned directly by the encoder.
    :type shift: numpy.ndarray
    :param norm: Per-feature normalization values used during encoding to scale the signal to [0, 1].
                 If ``None``, no amplitude rescaling is applied and only the shift is restored.
                 This is the ``norm`` value returned directly by the encoder.
    :type norm: numpy.ndarray | None
    :return: A numpy array representing the reconstructed continuous signal approximation.
    :rtype: numpy.ndarray
    :raises ValueError: If the input spike train is empty.

    """

    # Check for empty spike train
    if len(spikes) == 0:
        raise ValueError("Spike train cannot be empty.")

    # Ensure 2D processing (T, F)
    if spikes.ndim == 1:
        spikes = spikes.reshape(-1, 1)

    T, F = spikes.shape
    signal = np.zeros((T, F))

    for f in range(F):
        x = lfilter(fir_bank[:, f], 1, spikes[:, f], axis=0)
        if norm is not None:
            signal[:, f] = x * norm[f] + shift[f]
        else:
            signal[:, f] = x + shift[f]

    return signal
