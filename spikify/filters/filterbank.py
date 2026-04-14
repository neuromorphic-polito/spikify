import numpy as np
from scipy.signal import lfilter, butter, gammatone, sosfilt
from typing import Literal
from abc import ABC


class FilterBank(ABC):
    """
    A filter bank for decomposing signals into frequency components.

    This class decomposes input signals into frequency components using filter banks.
    Supported filter types are Butterworth, Gammatone, and second-order section (SOS) filters.
    The filter bank automatically computes center frequencies and frequency bands for each channel.

    **Code Example:**

    .. code-block:: python

        import numpy as np
        from spikify.filters import FilterBank
        fs = 1000
        signal = np.random.randn(1000)
        filterbank = FilterBank(fs=fs, channels=4, f_min=100, f_max=800, order=2)
        freq_components = filterbank.decompose(signal)

    .. doctest::
        :hide:

        >>> import numpy as np
        >>> from spikify.filters.filterbank import FilterBank
        >>> fs = 1000
        >>> signal = np.random.randn(1000)
        >>> filterbank = FilterBank(fs=fs, channels=4, f_min=100, f_max=800, order=2)
        >>> freq_components = filterbank.decompose(signal)
        >>> freq_components.shape
        (1000, 4, 1)

    :param fs: Sampling frequency of the input signal.
    :type fs: float
    :param channels: Number of filter channels.
    :type channels: int
    :param f_min: Minimum frequency for the filter bank.
    :type f_min: float
    :param f_max: Maximum frequency for the filter bank.
    :type f_max: float
    :param order: Order of the filters.
    :type order: int
    :param filter_type: Type of filter ('butterworth', 'gammatone', 'sos').
    :type filter_type: str
    :param kwargs: Additional filter parameters.
    :type kwargs: dict
    :raises ValueError: If filter_type is not supported.

    """

    def __init__(
        self,
        fs: float,
        channels: int,
        f_min: float,
        f_max: float,
        order: int,
        filter_type: Literal["butterworth", "gammatone", "sos"] = "butterworth",
        **kwargs,
    ):
        """Constructor method."""
        super().__init__()
        self.fs = fs
        self.filter_type = filter_type.lower()
        self.order = order
        self.n_channels = channels
        octave = (self.n_channels - 0.5) * np.log10(2) / np.log10(f_max / f_min)
        self.freq_centers = [f_min * (2 ** (ch / octave)) for ch in range(self.n_channels)]
        self.freq_poles = np.array(
            [(freq * (2 ** (-1 / (2 * octave))), (freq * (2 ** (1 / (2 * octave))))) for freq in self.freq_centers]
        )
        self.freq_poles[-1, 1] = self.fs / 2 * 0.99999

        self._build_filters(**kwargs)

    def _build_filters(self, **kwargs):
        """
        Build filter coefficients for all channels in the filter bank.

        This internal method constructs the filter coefficients for each channel based on the selected filter type and
        channel frequency bands or centers.

        :param kwargs: Additional filter parameters for specific filter types.
        :type kwargs: dict
        :raises ValueError: If filter type is not supported.

        """
        self.filter_coeffs = []
        self.channel_frequencies = []

        match self.filter_type:

            case "butterworth":
                for low_freq, high_freq in self.freq_poles:
                    num, den = butter(N=self.order, Wn=[low_freq, high_freq], btype="band", fs=self.fs, **kwargs)
                    self.filter_coeffs.append((num, den))
                    self.channel_frequencies.append((low_freq, high_freq))

            case "gammatone":
                for freq in self.freq_centers:
                    num, den = gammatone(order=self.order, freq=freq, ftype="fir", fs=self.fs, **kwargs)
                    self.filter_coeffs.append((num, den))
                    self.channel_frequencies.append(freq)

            case "sos":
                for low_freq, high_freq in self.freq_poles:
                    sos = butter(
                        N=self.order, Wn=[low_freq, high_freq], btype="band", output="sos", fs=self.fs, **kwargs
                    )
                    self.filter_coeffs.append(sos)
                    self.channel_frequencies.append((low_freq, high_freq))

            case _:
                raise ValueError(f"Filter {self.filter_type} is not supported")

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        Decompose input signal into frequency components using the filter bank.

        This method applies each filter in the bank to the input signal and returns the filtered outputs for all
        channels.

        :param signal: Input signal to be decomposed. Should be a 1D or 2D numpy array. If 2D, shape should be
            (timestamps, features).
        :type signal: numpy.ndarray
        :return: Array of filtered signals with shape (timestamps, channels, features).
        :rtype: numpy.ndarray

        """
        # Ensure 2D processing (T, F)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        T, F = signal.shape
        n_channels = len(self.filter_coeffs)

        # Initialize output
        freq_components = np.zeros((T, n_channels, F))

        for ch in range(n_channels):
            filter_coeffs = self.filter_coeffs[ch]

            if self.filter_type == "sos":
                # Use sosfilt for second-order sections
                for feat in range(F):
                    freq_components[:, ch, feat] = sosfilt(filter_coeffs, signal[:, feat])
            else:
                # Use lfilter for b,a coefficients
                for feat in range(F):
                    num, den = filter_coeffs
                    freq_components[:, ch, feat] = lfilter(num, den, signal[:, feat])

        return freq_components

    @property
    def center_frequencies(self) -> np.ndarray:
        """
        Center frequencies for each filter channel.

        This property returns the center frequency for each channel in the filter bank.

        :return: Array of center frequencies.
        :rtype: numpy.ndarray

        """
        return np.array([np.mean(freqs) for freqs in self.channel_frequencies])
