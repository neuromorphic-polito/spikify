import unittest
import numpy as np
from spikify.encoding.temporal.contrast.threshold_based_algorithm import threshold_based_representation


class TestThresholdBasedRepresentation(unittest.TestCase):
    """Tests for the threshold_based_representation function."""

    def test_basic_functionality(self):
        """Test the function with a basic signal where spikes should be generated based on threshold logic."""
        signal = np.array([0, 1, 2, 3, 2, 1, 0])
        factor = 1.0
        expected_spikes = np.array([0, 0, 0, 0, 0, 0, 0])
        result = threshold_based_representation(signal, factor)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_empty_signal(self):
        """Ensure the function raises an error with an empty signal."""
        signal = np.array([])
        factor = 1.0
        with self.assertRaises(ValueError):
            threshold_based_representation(signal, factor)

    def test_no_variation(self):
        """Test the function with a signal that has no variation; it should produce no spikes."""
        signal = np.array([1, 1, 1, 1, 1])
        factor = 1.0
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = threshold_based_representation(signal, factor)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_signal_with_noise(self):
        """Test the function's robustness against random noise in the signal."""
        np.random.seed(0)
        signal = np.array([0, 1, 2, 3, 2, 1, 0]) + np.random.randn(7) * 0.1
        factor = 0.5
        result = threshold_based_representation(signal, factor)
        self.assertTrue(np.any(result))  # Expect some spikes

    def test_large_signal(self):
        """Test the function's performance and correctness on a large signal."""
        signal = np.random.randn(1000) * 10  # Random signal with mean 0 and standard deviation 10
        factor = 1.0
        result = threshold_based_representation(signal, factor)
        self.assertEqual(len(result), len(signal))

    def test_varying_factor(self):
        """Test the function with different threshold factors to see varying spike density."""
        signal = np.array([0, 1, 2, 3, 2, 1, 0])

        # Higher threshold factor, less sensitivity
        factor_high = 2.0
        expected_spikes_high = np.array([0, 0, 0, 0, 0, 0, 0])
        result_high = threshold_based_representation(signal, factor_high)
        np.testing.assert_array_equal(result_high, expected_spikes_high)

        # Lower threshold factor, more sensitivity
        factor_low = 0.1
        expected_spikes_low = np.array([1, 1, 1, 1, -1, -1, -1])
        result_low = threshold_based_representation(signal, factor_low)
        np.testing.assert_array_equal(result_low, expected_spikes_low)

    def test_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = [0.5, 0.3]
        encoded_signal = threshold_based_representation(signal, threshold)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1 = threshold_based_representation(signal_f1, threshold[0])
        encoded_signal_f2 = threshold_based_representation(signal_f2, threshold[1])
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)
