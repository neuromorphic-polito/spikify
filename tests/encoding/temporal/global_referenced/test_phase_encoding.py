import unittest
import numpy as np
from spikify.encoding.temporal.global_referenced.phase_encoding_algorithm import (
    phase_encoding,
)  # Adjust this import based on your actual module structure


class TestPhaseEncoding(unittest.TestCase):
    """Tests phase_encoding function."""

    def test_basic_functionality(self):
        """Test the function's ability to encode a basic signal into phase encoding."""

        signal = np.array([0, 1, 2, 3, 4, 5, 6, 3, 2, 1])
        num_bits = 2
        result = phase_encoding(signal, num_bits)

        # Check the length of the result matches the expected length
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

        # Check that the resulting spike array contains only 0s and 1s
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

        # Ensure the result length is as expected
        self.assertEqual(len(result), len(signal))
