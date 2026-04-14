from spikify.encoders.temporal.deconvolution.modified_hough_spiker_algorithm import modified_hough_spiker
import unittest
import numpy as np


class TestModifiedHoughSpikerAlgorithm(unittest.TestCase):
    """Tests modified_hough_spiker function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes when the signal contains patterns that match the filter window
        and threshold conditions."""

        signal = np.array([0, 1.5, 2, 3, 4, 5, 6, 3, 2, 1, 0])
        window_length = 3
        threshold = 0.3
        cutoff = 0.2
        expected_spikes = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        result, _, _, _ = modified_hough_spiker(signal, window_length, cutoff, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_empty_signal(self):
        """Test the function with an empty signal."""

        signal = np.array([])
        window_length = 0
        cutoff = 0.2
        threshold = 1.0
        with self.assertRaises(ValueError):
            modified_hough_spiker(signal, window_length, cutoff, threshold)

    def test_filter_window_greater_than_signal_length(self):
        """Ensure the function raises an appropriate error when the filter window size is greater than the signal
        length."""

        signal = np.array([0, 1, 2, 3, 4, 5])
        window_length = 7
        cutoff = 0.2
        threshold = 1.0
        with self.assertRaises(ValueError):
            modified_hough_spiker(signal, window_length, cutoff, threshold)

    def test_no_matching_pattern(self):
        """Ensure the function correctly identifies when there are no generated spikes when the signal that lacks any
        pattern matching."""

        signal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        window_length = 5
        cutoff = 0.2
        threshold = -0.1
        expected_spikes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result, _, _, _ = modified_hough_spiker(signal, window_length, cutoff, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_single_point_spike(self):
        """Test the function’s ability to detect a spike when the signal contains a single sharp value."""

        signal = np.array([0, 0, 5, 0, 0])
        window_length = 1
        cutoff = 0.1
        threshold = 0.1
        expected_spikes = np.array([0, 0, 1, 0, 0])
        result, _, _, _ = modified_hough_spiker(signal, window_length, cutoff, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_varying_filter_window_size(self):
        """Ensure the function works correctly with different filter window sizes."""

        # Test case for filter window size of 3
        signal = np.array([0.1, 1.2, 2.3, 2.4, 3.2, 3.1, 2.3, 1.2, 0.1, 0.5, 0.4])
        cutoff = 0.1
        threshold = 0.1
        window_length_3 = 3
        expected_spikes_3 = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
        result_3, _, _, _ = modified_hough_spiker(signal, window_length_3, cutoff, threshold)
        np.testing.assert_array_equal(result_3, expected_spikes_3)

        # Test case for filter window size of 5
        window_length_5 = 5
        expected_spikes_5 = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1])
        result_5, _, _, _ = modified_hough_spiker(signal, window_length_5, cutoff, threshold)
        np.testing.assert_array_equal(result_5, expected_spikes_5)

        # Test case for filter window size of 7
        window_length_7 = 7
        expected_spikes_7 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        result_7, _, _, _ = modified_hough_spiker(signal, window_length_7, cutoff, threshold)
        np.testing.assert_array_equal(result_7, expected_spikes_7)

    def test_large_signal(self):
        """Test the function’s performance and correctness on a large signal."""

        signal = np.random.randn(10000)
        window_length = 10
        cutoff = 0.1
        threshold = 1.0
        result, _, _, _ = modified_hough_spiker(signal, window_length, cutoff, threshold)
        self.assertEqual(len(result), len(signal))

    def test_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        window_length = 3
        cutoff = 0.1
        threshold = [3.0, 5.0]
        encoded_signal, _, _, _ = modified_hough_spiker(signal, window_length, cutoff, threshold)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1, _, _, _ = modified_hough_spiker(signal_f1, window_length, cutoff, threshold[0])
        encoded_signal_f2, _, _, _ = modified_hough_spiker(signal_f2, window_length, cutoff, threshold[1])
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)

    def test_threshold_dims_different_from_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        window_length = 3
        cutoff = 0.2
        threshold = [0.1, 0.3, 0.4]
        with self.assertRaises(ValueError):
            modified_hough_spiker(signal, window_length, cutoff, threshold)

    def test_threshold_dimension(self):
        """Test that the function raises TypeError when threshold is of invalid dimension."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = np.array([[0.1, 0.2], [0.3, 0.4]])
        window_length = 3
        cutoff = 0.2
        with self.assertRaises(TypeError):
            modified_hough_spiker(signal, window_length, cutoff, threshold)

    def test_signal_with_negative_values(self):
        """Test the function with a signal containing negative values."""

        signal = np.array([-0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        window_length = 3
        cutoff = 0.4
        threshold = 0.1
        _, shift, norm, fir_coeff = modified_hough_spiker(signal, window_length, cutoff, threshold)

        expected_shift = np.array([-0.1], dtype=np.float32)
        expected_norm = np.array([1.0], dtype=np.float32)
        expected_fir_sum = 1.0

        np.testing.assert_array_equal(shift, expected_shift)
        np.testing.assert_array_equal(norm, expected_norm)
        self.assertAlmostEqual(fir_coeff.sum(), expected_fir_sum)

    def test_signal_with_positive_values_lower_than_one(self):
        """Test the function with a signal containing positive values lower than one."""

        signal = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.2], dtype=np.float32)
        window_length = 3
        cutoff = 0.4
        threshold = 0.1
        _, shift, norm, fir_coeff = modified_hough_spiker(signal, window_length, cutoff, threshold)

        expected_shift = np.array([0.0], dtype=np.float32)
        expected_norm = np.array([1.0], dtype=np.float32)
        expected_fir_sum = 1.0

        np.testing.assert_array_equal(shift, expected_shift)
        np.testing.assert_array_equal(norm, expected_norm)
        self.assertAlmostEqual(fir_coeff.sum(), expected_fir_sum)

    def test_signal_with_positive_values_greater_than_one(self):
        """Test the function with a signal containing positive values greater than one."""

        signal = np.array([0.1, 0.3, 0.5, 0.7, 1.9, 1.6, 1.4, 0.2], dtype=np.float32)
        window_length = 3
        cutoff = 0.4
        threshold = 0.1
        _, shift, norm, fir_coeff = modified_hough_spiker(signal, window_length, cutoff, threshold)

        expected_shift = np.array([0.0], dtype=np.float32)
        expected_norm = np.array([1.9], dtype=np.float32)
        expected_fir_sum = 1.0

        np.testing.assert_array_equal(shift, expected_shift)
        np.testing.assert_array_equal(norm, expected_norm)
        self.assertAlmostEqual(fir_coeff.sum(), expected_fir_sum)
