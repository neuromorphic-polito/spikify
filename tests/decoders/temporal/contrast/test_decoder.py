import unittest
import numpy as np
from spikify.decoders.temporal.contrast.decoder_algorithm import contrast_decoder


class TestContrastDecoderInput(unittest.TestCase):

    def test_empty_spike_train_raises(self):
        with self.assertRaises(ValueError):
            contrast_decoder(np.array([]), threshold=0.2, start_point=0.1)

    def test_invalid_threshold_dimension_raises(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        with self.assertRaises(TypeError):
            contrast_decoder(spikes, threshold=np.array([[0.2, 0.1], [0.3, 0.4]]), start_point=0.1)

    def test_invalid_start_point_dimension_raises(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        with self.assertRaises(TypeError):
            contrast_decoder(spikes, threshold=0.2, start_point=np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_threshold_size_mismatch_raises(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        with self.assertRaises(ValueError):
            contrast_decoder(spikes, threshold=[0.2, 0.3, 0.4], start_point=0.1)

    def test_start_point_size_mismatch_raises(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        with self.assertRaises(ValueError):
            contrast_decoder(spikes, threshold=0.2, start_point=[0.1, 0.2, 0.3])

    def test_output_is_ndarray(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        self.assertIsInstance(result, np.ndarray)

    def test_1d_input_returns_1d_output(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        self.assertEqual(result.ndim, 1)

    def test_2d_input_returns_2d_output(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        self.assertEqual(result.ndim, 2)

    def test_output_length_matches_input(self):
        spikes = np.array([0, 1, -1, 0, 1], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        self.assertEqual(len(result), len(spikes))

    def test_output_shape_matches_2d_input(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        self.assertEqual(result.shape, spikes.shape)

    def test_all_zeros_returns_constant_signal(self):
        spikes = np.array([0, 0, 0, 0], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.5)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_positive_spikes_increments(self):
        spikes = np.array([0, 1, 1, 1], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        expected = np.array([0.1, 0.3, 0.5, 0.7])
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_negative_spikes_decrements(self):
        spikes = np.array([0, -1, -1, -1], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.7)
        expected = np.array([0.7, 0.5, 0.3, 0.1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_spikes_reconstruction(self):
        spikes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.1)
        expected = np.array([0.1, 0.1, 0.1, 0.3, 0.5, 0.7])
        np.testing.assert_array_almost_equal(result, expected)

    def test_alternating_spikes_returns_to_start(self):
        spikes = np.array([0, 1, -1, 1, -1], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.2, start_point=0.5)
        expected = np.array([0.5, 0.7, 0.5, 0.7, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scalar_threshold_and_start_point(self):
        spikes = np.array([0, 1, 0, -1], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=0.1, start_point=0.0)
        expected = np.array([0.0, 0.1, 0.1, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_list_threshold_and_start_point(self):
        spikes = np.array([[0, 0], [1, -1], [0, 1]], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=[0.2, 0.3], start_point=[0.1, 0.5])
        expected = np.array([[0.1, 0.5], [0.3, 0.2], [0.3, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ndarray_threshold_and_start_point(self):
        spikes = np.array([[0, 0], [1, 1]], dtype=np.int8)
        result = contrast_decoder(spikes, threshold=np.array([0.2, 0.4]), start_point=np.array([0.0, 0.0]))
        expected = np.array([[0.0, 0.0], [0.2, 0.4]])
        np.testing.assert_array_almost_equal(result, expected)
