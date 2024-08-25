import unittest
import numpy as np
from spikify.encoding.temporal.latency.burst_encoding_algorithm import burst_encoding


class TestBurstEncoding(unittest.TestCase):
    """Tests for the burst_encoding function."""

    def test_basic_functionality(self):
        """Ensure the function correctly encodes a simple signal."""
        signal = np.array([0, 0.5, 1, 1.5, 2, 9])
        n_max = 3
        t_min = 1
        t_max = 5
        length = 6
        expected_output_length = length
        result = burst_encoding(signal, n_max, t_min, t_max, length)
        self.assertEqual(len(result), expected_output_length)
        self.assertTrue(np.all(result <= 1))

    def test_empty_signal(self):
        """Test the function with an empty signal."""
        signal = np.array([])
        n_max = 5
        t_min = 1
        t_max = 10
        length = 10
        with self.assertRaises(ValueError):
            burst_encoding(signal, n_max, t_min, t_max, length)

    def test_invalid_length_multiple(self):
        """Ensure the function raises an error if the length is not a multiple of the signal length."""
        signal = np.array([0, 1, 2, 3])
        n_max = 5
        t_min = 1
        t_max = 10
        length = 5
        with self.assertRaises(ValueError):
            burst_encoding(signal, n_max, t_min, t_max, length)

    def test_signal_normalization(self):
        """Ensure the function normalizes the signal correctly."""
        signal = np.random.rand(252)
        n_max = 3
        t_min = 1
        t_max = 5
        length = 12
        result = burst_encoding(signal, n_max, t_min, t_max, length)
        self.assertTrue(np.all(result <= 1))

    def test_spike_number_calculation(self):
        """Test the calculation of spike numbers based on signal intensity."""
        np.random.seed(42)
        signal = np.random.rand(240)
        n_max = 4
        t_min = 2
        t_max = 6
        length = 16
        result = burst_encoding(signal, n_max, t_min, t_max, length)
        expected_spike_counts = 53
        actual_spike_counts = np.sum(result)
        self.assertEqual(actual_spike_counts, expected_spike_counts)

    def test_insufficient_length_for_spikes(self):
        """Test that the function raises an error if the length is insufficient for the spikes and ISI."""
        signal = np.array([1, 1])
        n_max = 5
        t_min = 1
        t_max = 5
        length = 3
        with self.assertRaises(ValueError):
            burst_encoding(signal, n_max, t_min, t_max, length)

    def test_large_signal(self):
        """Test the function’s performance on a large signal."""
        signal = np.random.rand(1000)
        n_max = 5
        t_min = 1
        t_max = 10
        length = 100
        result = burst_encoding(signal, n_max, t_min, t_max, length)
        self.assertEqual(len(result), len(signal))
        self.assertTrue(np.all(result <= 1))
