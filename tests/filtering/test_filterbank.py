import unittest
import numpy as np
from spikify.filtering import FilterBank


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

    def test_invalid_filter_type(self):
        """Test that an invalid filter type raises a ValueError."""
        with self.assertRaises(ValueError):
            FilterBank(
                fs=self.fs,
                channels=self.channels,
                f_min=self.f_min,
                f_max=self.f_max,
                filter_type="invalid_type",
                order=self.order,
            )
