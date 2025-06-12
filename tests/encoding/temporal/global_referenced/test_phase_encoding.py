import unittest
import numpy as np
from spikify.encoding.temporal.global_referenced.phase_encoding_algorithm import (
    phase_encoding,
)


class TestPhaseEncoding(unittest.TestCase):
    """Tests phase_encoding function."""

    def test_basic_functionality(self):
        """Test the function's ability to encode a basic signal into phase encoding."""

        signal = np.array([0, 1, 2, 3, 4, 5, 6, 3, 2, 1])
        num_bits = 2
        result = phase_encoding(signal, num_bits)
        self.assertEqual(len(result), len(signal))

    def test_empty_signal(self):
        """Test that an empty signal raises a ValueError."""

        signal = np.array([])
        num_bits = 3
        with self.assertRaises(ValueError):
            phase_encoding(signal, num_bits)

    def test_non_multiple_signal_length(self):
        """Test that a signal with length not a multiple of num_bits raises a ValueError."""

        signal = np.array([0, 1, 2, 3, 4])
        num_bits = 3
        with self.assertRaises(ValueError):
            phase_encoding(signal, num_bits)

    def test_normalization(self):
        """Test that the signal is normalized when the maximum value is greater than 0."""

        signal = np.array([1, 2, 3, 4])
        num_bits = 2
        result = phase_encoding(signal, num_bits)
        self.assertTrue(np.all(np.logical_or(result == 0, result == 1)))

    def test_single_value_signal(self):
        """Test the function with a signal that has all identical values."""

        signal = np.array([3, 3, 3, 3, 3, 3])
        num_bits = 2
        result = phase_encoding(signal, num_bits)
        self.assertTrue(np.all(result == 1))

    def test_large_signal(self):
        """Test the function with a large signal to ensure it performs correctly."""

        signal = np.random.randn(10000)
        num_bits = 2
        result = phase_encoding(signal, num_bits)
        self.assertEqual(len(result), len(signal))

    def test_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        num_bit = 2
        encoded_signal = phase_encoding(signal, num_bit)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1 = phase_encoding(signal_f1, num_bit)
        encoded_signal_f2 = phase_encoding(signal_f2, num_bit)
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)
