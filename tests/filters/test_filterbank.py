import unittest
import numpy as np
from spikify.filters import FilterBank


class TestFilterBank(unittest.TestCase):
    """Tests for the FilterBank class."""

    def setUp(self):
        """Set up common parameters for the tests."""
        self.fs = 8000
        self.channels = 4
        self.f_min = 100
        self.f_max = 800
        self.order = 2
        self.signal_length = 1000
        self.signal = np.random.randn(self.signal_length)

    def test_butterworth_filterbank_initialization(self):
        """Test initialization of Butterworth filter bank."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="butterworth",
            order=self.order,
        )
        self.assertEqual(filterbank.fs, self.fs)
        self.assertEqual(filterbank.n_channels, self.channels)
        self.assertEqual(filterbank.filter_type, "butterworth")

    def test_gammatone_filterbank_initialization(self):
        """Test initialization of Gammatone filter bank."""
        filterbank = FilterBank(
            fs=self.fs, channels=self.channels, f_min=self.f_min, f_max=self.f_max, filter_type="gammatone", order=1
        )
        self.assertEqual(filterbank.fs, self.fs)
        self.assertEqual(filterbank.n_channels, self.channels)
        self.assertEqual(filterbank.filter_type, "gammatone")

    def test_sos_filterbank_initialization(self):
        """Test initialization of SOS filter bank."""
        filterbank = FilterBank(
            fs=self.fs, channels=self.channels, f_min=self.f_min, f_max=self.f_max, filter_type="sos", order=2
        )
        self.assertEqual(filterbank.fs, self.fs)
        self.assertEqual(filterbank.n_channels, self.channels)
        self.assertEqual(filterbank.filter_type, "sos")

    def test_butterworth_decomposition(self):
        """Test decomposition of a signal using Butterworth filter bank."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="butterworth",
            order=self.order,
        )
        freq_components = filterbank.decompose(self.signal)
        self.assertEqual(freq_components.shape, (self.signal_length, self.channels, 1))

    def test_gammatone_decomposition(self):
        """Test decomposition of a signal using Gammatone filter bank."""
        filterbank = FilterBank(
            fs=self.fs, channels=self.channels, f_min=self.f_min, f_max=self.f_max, filter_type="gammatone", order=1
        )
        freq_components = filterbank.decompose(self.signal)
        self.assertEqual(freq_components.shape, (self.signal_length, self.channels, 1))

    def test_sos_decomposition(self):
        """Test decomposition of a signal using SOS filter bank."""
        filterbank = FilterBank(
            fs=self.fs, channels=self.channels, f_min=self.f_min, f_max=self.f_max, filter_type="sos", order=2
        )
        freq_components = filterbank.decompose(self.signal)
        self.assertEqual(freq_components.shape, (self.signal_length, self.channels, 1))

    def test_signal_multiple_shape_decomposition(self):
        """Test that decomposing a too short signal raises a ValueError."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="butterworth",
            order=self.order,
        )
        signal = np.random.randn(10, 5, 3)
        with self.assertRaises(ValueError):
            filterbank.decompose(signal)

    def test_signal_with_multiple_features(self):
        """Test decomposition of a multi-feature signal."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="butterworth",
            order=self.order,
        )
        multi_feature_signal = np.random.randn(self.signal_length, 3)
        freq_components = filterbank.decompose(multi_feature_signal)
        self.assertEqual(freq_components.shape, (self.signal_length, self.channels, 3))

    def test_unsupported_filter(self):

        with self.assertRaises(ValueError):
            FilterBank(
                fs=self.fs,
                channels=self.channels,
                f_min=self.f_min,
                f_max=self.f_max,
                filter_type="gammachirp",
                order=self.order,
            )

    def test_center_frequencies_butterworth(self):
        """Test that center frequencies are computed correctly."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="butterworth",
            order=self.order,
        )

        octave = (self.channels - 0.5) * np.log10(2) / np.log10(self.f_max / self.f_min)
        freq_centers = np.array([self.f_min * (2 ** (ch / octave)) for ch in range(self.channels)])
        freq_poles = np.array(
            [(freq * (2 ** (-1 / (2 * octave))), (freq * (2 ** (1 / (2 * octave))))) for freq in freq_centers]
        )
        freq_poles[-1, 1] = self.fs / 2 * 0.99999

        expected_freq_centers = np.array([np.mean(freqs) for freqs in freq_poles])
        np.testing.assert_array_almost_equal(filterbank.center_frequencies, expected_freq_centers)

    def test_center_frequencies_gammatone(self):
        """Test that center frequencies are computed correctly."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="gammatone",
            order=self.order,
        )

        octave = (self.channels - 0.5) * np.log10(2) / np.log10(self.f_max / self.f_min)
        expected_freq_centers = np.array([self.f_min * (2 ** (ch / octave)) for ch in range(self.channels)])
        np.testing.assert_array_almost_equal(filterbank.center_frequencies, expected_freq_centers)

    def test_center_frequencies_sos(self):
        """Test that center frequencies are computed correctly."""
        filterbank = FilterBank(
            fs=self.fs,
            channels=self.channels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_type="sos",
            order=self.order,
        )

        octave = (self.channels - 0.5) * np.log10(2) / np.log10(self.f_max / self.f_min)
        freq_centers = np.array([self.f_min * (2 ** (ch / octave)) for ch in range(self.channels)])
        freq_poles = np.array(
            [(freq * (2 ** (-1 / (2 * octave))), (freq * (2 ** (1 / (2 * octave))))) for freq in freq_centers]
        )
        freq_poles[-1, 1] = self.fs / 2 * 0.99999

        expected_freq_centers = np.array([np.mean(freqs) for freqs in freq_poles])
        np.testing.assert_array_almost_equal(filterbank.center_frequencies, expected_freq_centers)
