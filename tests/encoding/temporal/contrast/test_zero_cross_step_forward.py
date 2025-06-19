import unittest
import numpy as np
from spikify.encoding.temporal.contrast.zero_cross_step_forward_algorithm import zero_cross_step_forward


class TestZeroCrossStepForward(unittest.TestCase):
    """Tests for the zero_cross_step_forward function."""

    def test_basic_functionality(self):
        """Test the function with a basic signal where some values exceed the threshold."""
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        threshold = 5.0
        expected_spikes = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        result = zero_cross_step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_empty_signal(self):
        """Ensure the function raises an error with an empty signal."""
        signal = np.array([])
        threshold = 5.0
        with self.assertRaises(ValueError):
            zero_cross_step_forward(signal, threshold)

    def test_all_below_threshold(self):
        """Test the function with a signal where all values are below the threshold."""
        signal = np.array([0, 1, 2, 3, 4])
        threshold = 5.0
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = zero_cross_step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_all_above_threshold(self):
        """Test the function with a signal where all values are above the threshold."""
        signal = np.array([6, 7, 8, 9, 10])
        threshold = 5.0
        expected_spikes = np.array([1, 1, 1, 1, 1])
        result = zero_cross_step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_signal_with_negative_values(self):
        """Test the function with a signal containing negative values."""
        signal = np.array([-1, -2, -3, 4, 5, -6, 7])
        threshold = 5.0
        expected_spikes = np.array([0, 0, 0, 0, 0, 0, 1])
        result = zero_cross_step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_no_spikes(self):
        """Test the function with a signal that has no positive values above the threshold."""
        signal = np.array([0, 0, 0, 0, 0])
        threshold = 1.0
        expected_spikes = np.array([0, 0, 0, 0, 0])
        result = zero_cross_step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_large_signal(self):
        """Test the function's performance and correctness on a large signal."""
        signal = np.random.randint(-10, 20, size=1000)  # Random signal with values between -10 and 20
        threshold = 10.0
        result = zero_cross_step_forward(signal, threshold)
        self.assertEqual(len(result), len(signal))

    def test_threshold_at_zero(self):
        """Test the function when the threshold is set to zero."""
        signal = np.array([-1, 0, 1, 2, 3])
        threshold = 0.0
        expected_spikes = np.array([0, 0, 1, 1, 1])
        result = zero_cross_step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_shape_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = [0.5, 0.3]
        encoded_signal = zero_cross_step_forward(signal, threshold)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1 = zero_cross_step_forward(signal_f1, threshold[0])
        encoded_signal_f2 = zero_cross_step_forward(signal_f2, threshold[1])
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)

    def test_factor_type(self):
        """Test the function with an Integer factor."""
        signal = np.random.rand(10)
        threshold = 3
        with self.assertRaises(TypeError):
            zero_cross_step_forward(signal, threshold)

    def test_factors_type(self):
        """Test the function with a list of Integer factors."""
        signal = np.random.rand(10, 2)
        thresholds = [5, 1]
        with self.assertRaises(TypeError):
            zero_cross_step_forward(signal, thresholds)

    def test_factors_length(self):
        """Test the function with a list of factors with dimension different from feature."""
        signal = np.random.rand(10, 2)
        thresholds = [5.0, 1.0, 3.0]
        with self.assertRaises(ValueError):
            zero_cross_step_forward(signal, thresholds)
