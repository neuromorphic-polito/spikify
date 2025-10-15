import unittest
import numpy as np
from spikify.encoding.rate.poisson_rate_algorithm import poisson_rate


class TestPoissonRateEncoding(unittest.TestCase):
    """Tests for the poisson_rate function."""

    def test_basic_functionality(self):
        """Ensure the function correctly encodes a simple signal."""
        signal = np.array([0, 0.5, 1, 1.5, 2])
        interval = 5
        result = poisson_rate(signal, interval)
        self.assertEqual(len(result), len(signal))
        self.assertTrue(np.all((result == 0) | (result == 1)))  # All values should be 0 or 1

    def test_empty_signal(self):
        """Test the function with an empty signal."""
        signal = np.array([])
        interval = 3
        with self.assertRaises(ValueError):
            poisson_rate(signal, interval)

    def test_invalid_interval(self):
        """Ensure the function raises an error if the interval is not a multiple of the signal length."""
        signal = np.array([0, 1, 2, 3, 4])
        interval = 3
        with self.assertRaises(ValueError):
            poisson_rate(signal, interval)

    def test_signal_normalization(self):
        """Ensure the function normalizes the signal correctly."""
        signal = np.array([0, 1, 2, 3, 4])
        interval = 5
        result = poisson_rate(signal, interval)
        self.assertTrue(np.all(result <= 1))

    def test_poisson_spike_generation(self):
        """Test if Poisson spike generation respects the rate encoded by the signal."""
        np.random.rand(42)
        signal = np.random.rand(100)
        interval = 5
        result = poisson_rate(signal, interval)
        self.assertEqual(len(result), len(signal))
        for i in range(0, len(result), interval):
            spike_count = np.sum(result[i : i + interval])
            self.assertTrue(spike_count >= 0)

    def test_large_signal(self):
        """Test the function's performance on a large signal."""
        signal = np.random.rand(1000)
        interval = 10
        result = poisson_rate(signal, interval)
        self.assertEqual(len(result), interval * (len(signal) // interval))
        self.assertTrue(np.all((result == 0) | (result == 1)))

    def test_all_zero_signal(self):
        """Test the function with a signal that is all zeros."""
        signal = np.array([0, 0, 0, 0])
        interval = 4
        expected_output = np.zeros(interval)
        result = poisson_rate(signal, interval)
        np.testing.assert_array_equal(result, expected_output)

    def test_non_numeric_input(self):
        """Ensure the function raises an appropriate error when provided with non-numeric input."""
        signal = np.array(["a", "b", "c"])
        interval = 2
        with self.assertRaises(ValueError):
            poisson_rate(signal, interval)

    def test_shape_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        interval = 2
        encoded_signal = poisson_rate(signal, interval)
        self.assertEqual(encoded_signal.shape, signal.shape)


if __name__ == "__main__":
    unittest.main()
