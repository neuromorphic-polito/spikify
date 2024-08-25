import unittest
import numpy as np
from spikify.encoding.temporal.contrast.moving_window_algorithm import moving_window


class TestMovingWindow(unittest.TestCase):
    """Tests for the moving_window function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes using a moving window."""
        signal = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
        window_length = 3
        threshold = 1
        expected_spikes = np.array([0, 0, 0, 1, 1, 0, -1, -1, -1])
        result = moving_window(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_empty_signal(self):
        """Test the function with an empty signal."""
        signal = np.array([])
        window_length = 3
        threshold = 1
        with self.assertRaises(ValueError):
            moving_window(signal, window_length, threshold)

    def test_window_length_greater_than_signal(self):
        """Ensure the function works when the window length is greater than the signal length."""
        signal = np.array([1, 2, 3])
        window_length = 5
        threshold = 1
        expected_spikes = np.array([0, 0, 0])
        result = moving_window(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_all_zero_signal(self):
        """Test the function with a signal that is all zeros."""
        signal = np.array([0, 0, 0, 0, 0])
        window_length = 3
        threshold = 1
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = moving_window(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_negative_threshold(self):
        """Ensure the function handles negative thresholds correctly."""
        signal = np.array([0, -1, -2, -1, 0, 1, 2])
        window_length = 3
        threshold = -1
        expected_spikes = np.array([1, 1, -1, 1, 1, 1, 1])
        result = moving_window(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_large_signal(self):
        """Test the function's performance and correctness on a large signal."""
        signal = np.random.randn(1000)
        window_length = 50
        threshold = 0.5
        result = moving_window(signal, window_length, threshold)
        self.assertEqual(len(result), len(signal))

    def test_no_spikes(self):
        """Test the function with a signal that should produce no spikes."""
        signal = np.array([1, 1, 1, 1, 1])
        window_length = 3
        threshold = 1
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = moving_window(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_variable_window_length(self):
        """Ensure the function works correctly with different window lengths."""
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0])

        # Test case for window length of 3
        window_length_3 = 3
        threshold_3 = 1
        expected_spikes_3 = np.array([0, 0, 0, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1])
        result_3 = moving_window(signal, window_length_3, threshold_3)
        np.testing.assert_array_equal(result_3, expected_spikes_3)

        # Test case for window length of 5
        window_length_5 = 5
        threshold_5 = 1
        expected_spikes_5 = np.array([-1, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1, -1, -1])
        result_5 = moving_window(signal, window_length_5, threshold_5)
        np.testing.assert_array_equal(result_5, expected_spikes_5)
