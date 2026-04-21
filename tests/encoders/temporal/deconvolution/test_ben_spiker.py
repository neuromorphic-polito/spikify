from spikify.encoders.temporal.deconvolution.bens_spiker_algorithm import bens_spiker
import unittest
import numpy as np


class TestBenSpikerAlgorithm(unittest.TestCase):
    """Tests ben_spiker function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes when the signal contains patterns that match the filter window
        and threshold conditions."""

        signal = np.array([0.3, 1.5, 2.8, 3.4, 4.6, 5.2, 6.8, 3.5, 2.4, 1.3, 0], dtype=np.float32)
        window_length = 5
        threshold = 3.0
        cutoff = 0.2
        expected_spikes = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        result, _, _ = bens_spiker(signal, window_length, cutoff, threshold)
        result = result.flatten()
        np.testing.assert_array_equal(result, expected_spikes)

    def test_threshold_sensitivity(self):
        """Verify that the function respects the threshold value by changing the threshold and observing the output."""

        signal = np.array([0, 1, 2, 3, 5, 3, 2, 1, 0], dtype=np.float32)
        window_length = 5
        threshold_low = 1.0
        threshold_high = 10.0
        cutoff = 0.2
        result_low, _, _ = bens_spiker(signal, window_length, cutoff, threshold_low)
        result_high, _, _ = bens_spiker(signal, window_length, cutoff, threshold_high)
        self.assertTrue(np.any(result_low))
        self.assertFalse(np.any(result_high))

    def test_empty_signal(self):
        """Test the function with an empty signal."""

        signal = np.array([])
        window_length = 0
        cutoff = 0.2
        threshold = 1.0
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, cutoff, threshold)

    def test_filter_window_greater_than_signal_length(self):
        """Ensure the function raises an appropriate error when the filter window size is greater than the signal
        length."""

        signal = np.array([0, 1, 2, 3, 4, 5])
        window_length = 7
        threshold = 1.0
        cutoff = 0.2
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, cutoff, threshold)

    def test_threshold_dims_different_from_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        window_length = 3
        cutoff = 0.2
        threshold = [0.1, 0.3, 0.4]
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, cutoff, threshold)

    def test_threshold_dimension(self):
        """Test that the function raises TypeError when threshold is of invalid dimension."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = np.array([[0.1, 0.2], [0.3, 0.4]])
        window_length = 3
        cutoff = 0.2
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, cutoff, threshold)

    def test_no_matching_pattern(self):
        """Ensure the function correctly identifies when there are no generated spikes when the signal that lacks any
        pattern matching."""

        signal = np.array([0, 0, 0, 0, 0])
        window_length = 3
        threshold = 1.0
        cutoff = 0.2
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result, _, _ = bens_spiker(signal, window_length, cutoff, threshold)
        result = result.flatten()
        np.testing.assert_array_equal(result, expected_spikes)

    def test_varying_filter_window_size(self):
        """Ensure the function works correctly with different filter window sizes."""

        # Test case for filter window size of 3
        signal = np.array([0, 1, 2, 3, 4, 5, 2, 1, 0], dtype=np.float32)
        threshold = 1.0
        cutoff = 0.2

        window_length_3 = 3
        expected_spikes_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        result_3, _, _ = bens_spiker(signal, window_length_3, cutoff, threshold)
        result_3 = result_3.flatten()
        np.testing.assert_array_equal(result_3, expected_spikes_3)

        # Test case for filter window size of 5
        window_length_5 = 5
        expected_spikes_5 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0])
        result_5, _, _ = bens_spiker(signal, window_length_5, cutoff, threshold)
        result_5 = result_5.flatten()
        np.testing.assert_array_equal(result_5, expected_spikes_5)

        # Test case for filter window size of 7

        window_length_7 = 7
        expected_spikes_7 = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0])
        result_7, _, _ = bens_spiker(signal, window_length_7, cutoff, threshold)
        result_7 = result_7.flatten()
        np.testing.assert_array_equal(result_7, expected_spikes_7)

    def test_large_signal(self):
        """Test the function’s performance and correctness on a large signal."""

        signal = np.random.randn(10000)
        window_length = 10
        threshold = 5.0
        cutoff = 0.2
        result, _, _ = bens_spiker(signal, window_length, cutoff, threshold)
        result = result.flatten()
        self.assertEqual(len(result), len(signal))

    def test_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = [0.5, 0.3]
        window_length = 3
        cutoff = 0.2
        encoded_signal, _, _ = bens_spiker(signal, window_length, cutoff, threshold)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1, _, _ = bens_spiker(signal_f1, window_length, cutoff, threshold[0])
        encoded_signal_f1 = encoded_signal_f1.flatten()
        encoded_signal_f2, _, _ = bens_spiker(signal_f2, window_length, cutoff, threshold[1])
        encoded_signal_f2 = encoded_signal_f2.flatten()
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)

    def test_list_threshold_differnt_features(self):
        """Test the function with a size window length major of signal length."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = [0.5, 0.3, 0.1]
        cutoff = 0.2
        window_length = 3
        with self.assertRaises(ValueError):
            bens_spiker(signal, window_length, cutoff, threshold, window_type="lanczos")

    def test_signal_with_negative_values(self):
        """Test the function with a signal containing negative values."""

        signal = np.array([-0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        window_length = 3
        cutoff = 0.4
        threshold = 0.95
        _, fir_coeff, shift = bens_spiker(signal, window_length, cutoff, threshold)

        expected_shift = np.array([-0.1], dtype=np.float32)
        expected_fir_sum = 1.0

        np.testing.assert_array_equal(shift, expected_shift)
        self.assertAlmostEqual(fir_coeff.sum(), expected_fir_sum)

    def test_signal_with_positive_values_lower_than_one(self):
        """Test the function with a signal containing positive values lower than one."""

        signal = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.2], dtype=np.float32)
        window_length = 3
        cutoff = 0.4
        threshold = 0.95
        _, fir_coeff, shift = bens_spiker(signal, window_length, cutoff, threshold)

        expected_shift = np.array([0.0], dtype=np.float32)
        expected_fir_sum = 1.0

        np.testing.assert_array_equal(shift, expected_shift)
        self.assertAlmostEqual(fir_coeff.sum(), expected_fir_sum)

    def test_signal_with_positive_values_greater_than_one(self):
        """Test the function with a signal containing positive values greater than one."""

        signal = np.array([0.1, 0.3, 0.5, 0.7, 1.9, 1.6, 1.4, 0.2], dtype=np.float32)
        window_length = 3
        cutoff = 0.4
        threshold = 0.95
        _, fir_coeff, shift = bens_spiker(signal, window_length, cutoff, threshold)

        expected_shift = np.array([0.0], dtype=np.float32)
        expected_fir_sum = 3.8

        np.testing.assert_array_equal(shift, expected_shift)
        self.assertAlmostEqual(fir_coeff.sum(), expected_fir_sum)
