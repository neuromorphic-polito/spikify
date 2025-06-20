import unittest
import numpy as np
from spikify.encoding.temporal.contrast.moving_window_algorithm import moving_window


class TestMovingWindow(unittest.TestCase):
    """Tests for the moving_window function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes using a moving window."""
        signal = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
        window_length = 3
        expected_spikes = np.array([0, 0, 0, 1, 1, 0, -1, -1, -1])
        result = moving_window(signal, window_length)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_empty_signal(self):
        """Test the function with an empty signal."""
        signal = np.array([])
        window_length = 3
        with self.assertRaises(ValueError):
            moving_window(signal, window_length)

    def test_window_length_greater_than_signal(self):
        """Ensure the function works when the window length is greater than the signal length."""
        signal = np.array([1, 2, 3])
        window_length = 5
        expected_spikes = np.array([0, 0, 0])
        result = moving_window(signal, window_length)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_all_zero_signal(self):
        """Test the function with a signal that is all zeros."""
        signal = np.array([0, 0, 0, 0, 0])
        window_length = 3
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = moving_window(signal, window_length)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_large_signal(self):
        """Test the function's performance and correctness on a large signal."""
        signal = np.random.randn(1000)
        window_length = 50
        result = moving_window(signal, window_length)
        self.assertEqual(len(result), len(signal))

    def test_no_spikes(self):
        """Test the function with a signal that should produce no spikes."""
        signal = np.array([1, 1, 1, 1, 1])
        window_length = 3
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = moving_window(signal, window_length)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_variable_window_length(self):
        """Ensure the function works correctly with different window lengths."""
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0])

        # Test case for window length of 3
        window_length_3 = 3
        expected_spikes_3 = np.array([0, 0, 0, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1])
        result_3 = moving_window(signal, window_length_3)
        np.testing.assert_array_equal(result_3, expected_spikes_3)

        # Test case for window length of 5
        window_length_5 = 5
        expected_spikes_5 = np.array([-1, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1, -1, -1])
        result_5 = moving_window(signal, window_length_5)
        np.testing.assert_array_equal(result_5, expected_spikes_5)

    def test_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        interval = 2
        encoded_signal = moving_window(signal, interval)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1 = moving_window(signal_f1, interval)
        encoded_signal_f2 = moving_window(signal_f2, interval)
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)
