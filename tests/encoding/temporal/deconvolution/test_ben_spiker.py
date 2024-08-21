from spikify.encoding.temporal.deconvolution.ben_spiker_algorithm import ben_spiker
import unittest
import numpy as np


class TestBenSpikerAlgorithm(unittest.TestCase):
    """Tests ben_spiker function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes when the signal contains patterns that match the filter
        window and threshold conditions."""

        signal = np.array([0, 1.5, 2, 3, 4, 5, 6, 3, 2, 1, 0])
        bsa_filter_window = 3
        bsa_threshold = 2
        expected_spikes = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        result = ben_spiker(signal, bsa_filter_window, bsa_threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_threshold_sensitivity(self):
        """Verify that the function respects the threshold value by changing the threshold and observing the output."""

        signal = np.array([0, 1, 2, 3, 5, 3, 2, 1, 0])
        bsa_filter_window = 3
        bsa_threshold_low = 1
        bsa_threshold_high = 5
        result_low = ben_spiker(signal, bsa_filter_window, bsa_threshold_low)
        result_high = ben_spiker(signal, bsa_filter_window, bsa_threshold_high)
        self.assertTrue(np.any(result_low))
        self.assertFalse(np.any(result_high))

    def test_empty_signal(self):
        """Test the function with an empty signal."""

        signal = np.array([])
        bsa_filter_window = 0
        bsa_threshold = 1
        with self.assertRaises(ValueError):
            ben_spiker(signal, bsa_filter_window, bsa_threshold)

    def test_filter_window_greater_than_signal_length(self):
        """Ensure the function raises an appropriate error when the filter window size is greater than the signal
        length."""

        signal = np.array([0, 1, 2, 3, 4, 5])
        bsa_filter_window = 6
        bsa_threshold = 1
        with self.assertRaises(ValueError):
            ben_spiker(signal, bsa_filter_window, bsa_threshold)

    def test_no_matching_pattern(self):
        """Ensure the function correctly identifies when there are no generated spikes when the signal that lacks any
        pattern matching."""

        signal = np.array([0, 0, 0, 0, 0])
        bsa_filter_window = 3
        bsa_threshold = 1
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = ben_spiker(signal, bsa_filter_window, bsa_threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_single_point_spike(self):
        """Test the function’s ability to detect a spike when the signal contains a single sharp value."""

        signal = np.array([0, 0, 5, 0, 0])
        bsa_filter_window = 1
        bsa_threshold = 1
        expected_spikes = np.array([0, 0, 1, 0, 0])
        result = ben_spiker(signal, bsa_filter_window, bsa_threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_varying_filter_window_size(self):
        """Ensure the function works correctly with different filter window sizes."""

        # Test case for filter window size of 3
        signal_3 = np.array([0, 1, 2, 3, 4, 5, 2, 1, 0])
        bsa_filter_window_3 = 3
        bsa_threshold_3 = 1
        expected_spikes_3 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
        result_3 = ben_spiker(signal_3, bsa_filter_window_3, bsa_threshold_3)
        np.testing.assert_array_equal(result_3, expected_spikes_3)

        # Test case for filter window size of 5
        signal_5 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
        bsa_filter_window_5 = 5
        bsa_threshold_5 = 1
        expected_spikes_5 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result_5 = ben_spiker(signal_5, bsa_filter_window_5, bsa_threshold_5)
        np.testing.assert_array_equal(result_5, expected_spikes_5)

        # Test case for filter window size of 7
        signal_7 = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0])
        bsa_filter_window_7 = 7
        bsa_threshold_7 = 1
        expected_spikes_7 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        result_7 = ben_spiker(signal_7, bsa_filter_window_7, bsa_threshold_7)
        np.testing.assert_array_equal(result_7, expected_spikes_7)

    def test_boundary_conditions(self):
        """Test how the function handles boundary conditions where the spikes generated occur at the beginning or
        end of the signal."""

        signal = np.array([5, 5, 5, 0, 0])
        bsa_filter_window = 3
        bsa_threshold = 1
        expected_spikes = np.array([1, 1, 0, 0, 0])
        result = ben_spiker(signal, bsa_filter_window, bsa_threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_large_signal(self):
        """Test the function’s performance and correctness on a large signal."""

        signal = np.random.randn(10000)
        bsa_filter_window = 10
        bsa_threshold = 5
        result = ben_spiker(signal, bsa_filter_window, bsa_threshold)
        self.assertEqual(len(result), len(signal))

    def test_non_numeric_input(self):
        """Ensure the function raises an appropriate error when provided with non-numeric input."""

        signal = np.array(["a", "b", "c"])
        bsa_filter_window = 2
        bsa_threshold = 1
        with self.assertRaises(TypeError):
            ben_spiker(signal, bsa_filter_window, bsa_threshold)

    def test_noise(self):
        """Test the function's robustness against random noise in the signal."""

        np.random.seed(0)
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 3, 2, 1, 0]) + np.random.randn(11)
        bsa_filter_window = 3
        bsa_threshold = 1
        result = ben_spiker(signal, bsa_filter_window, bsa_threshold)
        self.assertTrue(np.any(result))
