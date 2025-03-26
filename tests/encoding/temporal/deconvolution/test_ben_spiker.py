from spikify.encoding.temporal.deconvolution.bens_spiker_algorithm import bens_spiker
import unittest
import numpy as np


class TestBenSpikerAlgorithm(unittest.TestCase):
    """Tests ben_spiker function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes when the signal contains patterns that match the filter window
        and threshold conditions."""

        signal = np.array([0, 1.5, 2, 3, 4, 5, 6, 3, 2, 1, 0])
        window_length = 3
        threshold = 2
        expected_spikes = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        result = bens_spiker(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_threshold_sensitivity(self):
        """Verify that the function respects the threshold value by changing the threshold and observing the output."""

        signal = np.array([0, 1, 2, 3, 5, 3, 2, 1, 0])
        window_length = 3
        threshold_low = 1
        threshold_high = 5
        result_low = bens_spiker(signal, window_length, threshold_low)
        result_high = bens_spiker(signal, window_length, threshold_high)
        self.assertTrue(np.any(result_low))
        self.assertFalse(np.any(result_high))

    def test_empty_signal(self):
        """Test the function with an empty signal."""

        signal = np.array([])
        window_length = 0
        threshold = 1
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, threshold)

    def test_filter_window_greater_than_signal_length(self):
        """Ensure the function raises an appropriate error when the filter window size is greater than the signal
        length."""

        signal = np.array([0, 1, 2, 3, 4, 5])
        window_length = 7
        threshold = 1
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, threshold)

    def test_no_matching_pattern(self):
        """Ensure the function correctly identifies when there are no generated spikes when the signal that lacks any
        pattern matching."""

        signal = np.array([0, 0, 0, 0, 0])
        window_length = 3
        threshold = 1
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = bens_spiker(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_single_point_spike(self):
        """Test the function’s ability to detect a spike when the signal contains a single sharp value."""

        signal = np.array([0, 0, 5, 0, 0])
        window_length = 1
        threshold = 1
        expected_spikes = np.array([0, 0, 1, 0, 0])
        result = bens_spiker(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_varying_filter_window_size(self):
        """Ensure the function works correctly with different filter window sizes."""

        # Test case for filter window size of 3
        signal_3 = np.array([0, 1, 2, 3, 4, 5, 2, 1, 0])
        window_length_3 = 3
        threshold_3 = 1
        expected_spikes_3 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
        result_3 = bens_spiker(signal_3, window_length_3, threshold_3)
        np.testing.assert_array_equal(result_3, expected_spikes_3)

        # Test case for filter window size of 5
        signal_5 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
        window_length_5 = 5
        threshold_5 = 1
        expected_spikes_5 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result_5 = bens_spiker(signal_5, window_length_5, threshold_5)
        np.testing.assert_array_equal(result_5, expected_spikes_5)

        # Test case for filter window size of 7
        signal_7 = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0])
        window_length_7 = 7
        threshold_7 = 1
        expected_spikes_7 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        result_7 = bens_spiker(signal_7, window_length_7, threshold_7)
        np.testing.assert_array_equal(result_7, expected_spikes_7)

    def test_boundary_conditions(self):
        """Test how the function handles boundary conditions where the spikes generated occur at the beginning or end of
        the signal."""

        signal = np.array([5, 5, 5, 0, 0])
        window_length = 3
        threshold = 1
        expected_spikes = np.array([1, 1, 0, 0, 0])
        result = bens_spiker(signal, window_length, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_large_signal(self):
        """Test the function’s performance and correctness on a large signal."""

        signal = np.random.randn(10000)
        window_length = 10
        threshold = 5
        result = bens_spiker(signal, window_length, threshold)
        self.assertEqual(len(result), len(signal))

    def test_non_numeric_input(self):
        """Ensure the function raises an appropriate error when provided with non-numeric input."""

        signal = np.array(["a", "b", "c"])
        window_length = 2
        threshold = 1
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, threshold)

    def test_noise(self):
        """Test the function's robustness against random noise in the signal."""

        np.random.seed(0)
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 3, 2, 1, 0]) + np.random.randn(11)
        window_length = 3
        threshold = 1
        result = bens_spiker(signal, window_length, threshold)
        self.assertTrue(np.any(result))
