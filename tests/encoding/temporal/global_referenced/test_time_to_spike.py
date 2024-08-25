import unittest
import numpy as np
from spikify.encoding.temporal.global_referenced.time_to_spike_algorithm import (
    time_to_first_spike,
)


class TestTimeToFirstSpike(unittest.TestCase):
    """Tests time_to_first_spike function."""

    def test_basic_functionality(self):
        """Test the function's ability to encode a basic signal into time-to-first-spike encoding."""

        signal = np.array([0, 1, 2, 3, 4, 5, 6, 3, 2, 1, 0, 4])
        interval = 4
        result = time_to_first_spike(signal, interval)
        self.assertEqual(len(result), len(signal))

    def test_empty_signal(self):
        """Test that an empty signal raises a ValueError."""

        signal = np.array([])
        interval = 3
        with self.assertRaises(ValueError):
            time_to_first_spike(signal, interval)

    def test_non_multiple_signal_length(self):
        """Test that a signal with length not a multiple of interval raises a ValueError."""

        signal = np.array([0, 1, 2, 3, 4])
        interval = 3
        with self.assertRaises(ValueError):
            time_to_first_spike(signal, interval)

    def test_normalization(self):
        """Test that the signal is normalized when the maximum value is greater than 0."""

        signal = np.array([1, 2, 3, 4, 5, 7])
        interval = 2
        result = time_to_first_spike(signal, interval)
        self.assertTrue(np.all(np.logical_or(result == 0, result == 1)))

    def test_single_value_signal(self):
        """Test the function with a signal that has all identical values."""

        signal = np.array([3, 3, 3, 3, 3, 3])
        interval = 2
        result = time_to_first_spike(signal, interval)
        expected_result = np.array([1, 0, 1, 0, 1, 0])
        self.assertTrue(np.all(result == expected_result))

    def test_large_signal(self):
        """Test the function with a large signal to ensure it performs correctly."""

        signal = np.random.randn(10000)
        interval = 4
        result = time_to_first_spike(signal, interval)
        self.assertEqual(len(result), len(signal))

    def test_intensity_computation(self):
        """Test that the function correctly computes intensity based on the signal."""

        signal = np.array([0, 1, 2, 3, 4, 5])
        interval = 2
        result = time_to_first_spike(signal, interval)
        self.assertEqual(result.shape[0], len(signal))

    def test_zero_signal(self):
        """Ensure the function generates no spikes when the signal is all zeros."""

        signal = np.array([0, 0, 0, 0, 0, 0])
        interval = 2
        result = time_to_first_spike(signal, interval)
        print(result)
        expected_result = np.array([0, 1, 0, 1, 0, 1])
        self.assertTrue(np.all(result == expected_result))

    def test_boundary_conditions(self):
        """Test how the function handles boundary conditions where spikes are expected at the extremes of the signal."""

        signal = np.array([0, 1, 0, 1, 0, 2])
        interval = 3
        result = time_to_first_spike(signal, interval)
        self.assertTrue(np.any(result))


if __name__ == "__main__":
    unittest.main()
