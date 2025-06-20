import unittest
import numpy as np
from spikify.encoding.temporal.contrast.step_forward_algorithm import step_forward


class TestStepForward(unittest.TestCase):
    """Tests for the step_forward function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes based on the threshold logic."""
        signal = np.array([0, 2, 4, 6, 4, 2, 0])
        threshold = 2.0
        expected_spikes = np.array([0, 0, 1, 1, 0, 0, -1])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_empty_signal(self):
        """Test the function with an empty signal."""
        signal = np.array([])
        threshold = 2.0
        with self.assertRaises(ValueError):
            step_forward(signal, threshold)

    def test_no_spikes(self):
        """Test the function when no spikes should be generated (signal stays within the threshold)."""
        signal = np.array([1, 1.5, 1.8, 1.6, 1.4, 1.2])
        threshold = 2.0
        expected_spikes = np.array([0, 0, 0, 0, 0, 0])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_single_positive_spike(self):
        """Test the function’s ability to detect a single positive spike."""
        signal = np.array([0, 0, 5, 0, 0])
        threshold = 4.0
        expected_spikes = np.array([0, 0, 1, 0, 0])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_single_negative_spike(self):
        """Test the function’s ability to detect a single negative spike."""
        signal = np.array([0, 0, -5, 0, 0])
        threshold = 4.0
        expected_spikes = np.array([0, 0, -1, 0, 0])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_alternating_spikes(self):
        """Test the function with alternating spikes."""
        signal = np.array([0, 5, 0, -5, 0])
        threshold = 4.0
        expected_spikes = np.array([0, 1, 0, -1, 0])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_signal_with_noise(self):
        """Test the function with a noisy signal where spikes should still be detected."""
        signal = np.array([0, 0.5, 2.5, 1.5, 0, -1, -2.5, -1.5, 0])
        threshold = 2.0
        expected_spikes = np.array([0, 0, 1, 0, 0, -1, -1, 0, 0])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_large_signal(self):
        """Test the function's performance and correctness on a large signal."""
        signal = np.random.randn(1000) * 10  # Random signal with mean 0 and standard deviation 10
        threshold = 15.0
        result = step_forward(signal, threshold)
        self.assertEqual(len(result), len(signal))

    def test_boundary_conditions(self):
        """Test how the function handles boundary conditions where spikes occur at the start or end of the signal."""
        signal = np.array([10, 0, -10])
        threshold = 5.0
        expected_spikes = np.array([0, -1, -1])
        result = step_forward(signal, threshold)
        np.testing.assert_array_equal(result, expected_spikes)

    def test_with_multiple_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = [0.1, 0.3]
        encoded_signal = step_forward(signal, threshold)
        self.assertEqual(encoded_signal.shape, signal.shape)
        signal_f1 = signal[:, 0]
        signal_f2 = signal[:, 1]
        encoded_signal_f1 = step_forward(signal_f1, threshold[0])
        encoded_signal_f2 = step_forward(signal_f2, threshold[1])
        np.testing.assert_array_equal(encoded_signal[:, 0], encoded_signal_f1)
        np.testing.assert_array_equal(encoded_signal[:, 1], encoded_signal_f2)

    def test_features_list(self):
        """Test the function with a integer thrshold."""
        signal = np.array([0, 1, 2, 3, 2, 1, 0])
        thresholds = 1
        with self.assertRaises(TypeError):
            step_forward(signal, thresholds)

    def test_thrsholds_different_from_features(self):
        """Test the function with a signal containing multiple features."""
        np.random.seed(42)
        signal = np.random.rand(10, 2)
        threshold = [0.1, 0.3, 0.4]
        with self.assertRaises(ValueError):
            step_forward(signal, threshold)
