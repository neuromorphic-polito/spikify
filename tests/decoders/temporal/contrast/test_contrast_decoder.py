import unittest
import numpy as np
from spikify.decoders.temporal.contrast.decoder_algorithm import contrast_decoder
from spikify.encoders.temporal.contrast import (
    threshold_based_representation,
    step_forward,
    moving_window,
    zero_cross_step_forward,
)


class TestContrastDecoderInput(unittest.TestCase):

    def test_empty_spike_train_raises(self):
        with self.assertRaises(ValueError):
            contrast_decoder(np.array([]), thresholds=np.array([0.2]), start_point=0.1)

    def test_invalid_start_point_dimension_raises(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_start_point_size_mismatch_raises(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        with self.assertRaises(ValueError):
            contrast_decoder(spikes, thresholds=np.array([0.2, 0.3]), start_point=[0.1, 0.2, 0.3])

    def test_output_is_ndarray(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.1)
        self.assertIsInstance(result, np.ndarray)

    def test_1d_input_returns_2d_output(self):
        spikes = np.array([0, 1, -1, 0], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.1)
        self.assertEqual(result.ndim, 2)

    def test_2d_input_returns_2d_output(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2, 0.2]), start_point=0.1)
        self.assertEqual(result.ndim, 2)

    def test_output_length_matches_input(self):
        spikes = np.array([0, 1, -1, 0, 1], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.1)
        self.assertEqual(len(result), len(spikes))

    def test_output_shape_matches_2d_input(self):
        spikes = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2, 0.2]), start_point=0.1)
        self.assertEqual(result.shape, spikes.shape)

    def test_all_zeros_returns_constant_signal(self):
        spikes = np.array([0, 0, 0, 0], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.5)
        expected = np.array([0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_positive_spikes_increments(self):
        spikes = np.array([0, 1, 1, 1], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.1)
        expected = np.array([0.1, 0.3, 0.5, 0.7]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_negative_spikes_decrements(self):
        spikes = np.array([0, -1, -1, -1], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.7)
        expected = np.array([0.7, 0.5, 0.3, 0.1]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_spikes_reconstruction(self):
        spikes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.1)
        expected = np.array([0.1, 0.1, 0.1, 0.3, 0.5, 0.7]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_alternating_spikes_returns_to_start(self):
        spikes = np.array([0, 1, -1, 1, -1], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2]), start_point=0.5)
        expected = np.array([0.5, 0.7, 0.5, 0.7, 0.5]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scalar_start_point(self):
        spikes = np.array([0, 1, 0, -1], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.1]), start_point=0.0)
        expected = np.array([0.0, 0.1, 0.1, 0.0]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_feature_ndarray_threshold_and_list_start_point(self):
        spikes = np.array([[0, 0], [1, -1], [0, 1]], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2, 0.3]), start_point=[0.1, 0.5])
        expected = np.array([[0.1, 0.5], [0.3, 0.2], [0.3, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_feature_ndarray_threshold_and_ndarray_start_point(self):
        spikes = np.array([[0, 0], [1, 1]], dtype=np.int8)
        result = contrast_decoder(spikes, thresholds=np.array([0.2, 0.4]), start_point=np.array([0.0, 0.0]))
        expected = np.array([[0.0, 0.0], [0.2, 0.4]])
        np.testing.assert_array_almost_equal(result, expected)


class TestContrastDecoderRoundTripTBR(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6]).reshape(-1, 1)
        spikes, thresholds = threshold_based_representation(signal, factor=0.5)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        spikes, thresholds = threshold_based_representation(signal, factor=0.5)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_starts_at_signal_start_point(self):
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        spikes, thresholds = threshold_based_representation(signal, factor=0.5)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertAlmostEqual(result[0], signal[0])

    def test_round_trip_multi_feature(self):
        signal = np.array([[0.1, 0.5], [0.3, 0.4], [0.4, 0.6], [0.2, 0.3], [0.5, 0.7], [0.6, 0.8]])
        spikes, thresholds = threshold_based_representation(signal, factor=0.5)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_zero_factor_all_zeros(self):
        signal = np.array([0.5, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
        spikes, thresholds = threshold_based_representation(signal, factor=0.0)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        np.testing.assert_array_almost_equal(result, np.full_like(signal, signal[0]))


class TestContrastDecoderRoundTripSF(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6]).reshape(-1, 1)
        spikes, thresholds = step_forward(signal, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        spikes, thresholds = step_forward(signal, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_starts_at_signal_start_point(self):
        signal = np.array([0.1, 0.3, 0.4, 0.2, 0.5, 0.6])
        spikes, thresholds = step_forward(signal, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertAlmostEqual(result[0], signal[0])

    def test_round_trip_flat_signal_no_spikes(self):
        signal = np.array([0.5, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
        spikes, thresholds = step_forward(signal, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        np.testing.assert_array_almost_equal(result, np.full_like(signal, signal[0]))

    def test_round_trip_monotonic_increase(self):
        signal = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).reshape(-1, 1)
        spikes, thresholds = step_forward(signal, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)
        self.assertTrue(np.all(result >= signal[0]))

    def test_round_trip_multi_feature(self):
        signal = np.array([[0.1, 0.5], [0.3, 0.4], [0.4, 0.6], [0.2, 0.3], [0.5, 0.7], [0.6, 0.8]])
        spikes, thresholds = step_forward(signal, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)


class TestContrastDecoderRoundTripMW(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0]).reshape(-1, 1)
        spikes, thresholds = moving_window(signal, window_length=3, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        spikes, thresholds = moving_window(signal, window_length=3, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_starts_at_signal_start_point(self):
        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0])
        spikes, thresholds = moving_window(signal, window_length=3, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertAlmostEqual(result[0], signal[0])

    def test_round_trip_flat_signal_no_spikes(self):
        signal = np.array([0.5, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
        spikes, thresholds = moving_window(signal, window_length=3, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        np.testing.assert_array_almost_equal(result, np.full_like(signal, signal[0]))

    def test_round_trip_multi_feature(self):
        signal = np.array([[0.1, 0.5], [0.3, 0.4], [0.4, 0.6], [0.2, 0.3], [0.5, 0.7], [0.6, 0.8]])
        spikes, thresholds = moving_window(signal, window_length=3, threshold=0.2)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_window_length_one(self):
        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0]).reshape(-1, 1)
        spikes, thresholds = moving_window(signal, window_length=1, threshold=0.1)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)


class TestContrastDecoderRoundTripZCSF(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3]).reshape(-1, 1)
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_starts_at_signal_start_point(self):
        signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertAlmostEqual(result[0], signal[0])

    def test_round_trip_all_negative_signal_no_spikes(self):
        signal = np.array([-0.5, -0.3, -0.1, -0.8, -0.2]).reshape(-1, 1)
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        np.testing.assert_array_almost_equal(result, np.full_like(signal, signal[0]))

    def test_round_trip_all_below_threshold_no_spikes(self):
        signal = np.array([0.1, 0.2, 0.1, 0.3, 0.2]).reshape(-1, 1)
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        np.testing.assert_array_almost_equal(result, np.full_like(signal, signal[0]))

    def test_round_trip_only_positive_spikes(self):
        signal = np.array([-0.2, 0.1, 0.5, 0.0, 1.2, 0.3])
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        self.assertTrue(np.all(spikes >= 0))
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertTrue(np.all(np.diff(result) >= 0))

    def test_round_trip_multi_feature(self):
        signal = np.array([[-0.2, 0.5], [0.1, 0.0], [0.5, 1.2], [0.0, 0.3], [1.2, 0.8], [0.3, 0.6]])
        spikes, thresholds = zero_cross_step_forward(signal, threshold=0.4)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_high_threshold_no_spikes(self):
        signal = np.array([0.1, 0.3, 0.2, 0.5, 0.4]).reshape(-1, 1)
        spikes, thresholds = zero_cross_step_forward(signal, threshold=999.0)
        result = contrast_decoder(spikes, thresholds, start_point=signal[0])
        np.testing.assert_array_almost_equal(result, np.full_like(signal, signal[0]))
