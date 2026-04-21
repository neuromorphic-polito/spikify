import unittest
import numpy as np
from spikify.decoders.temporal.deconvolution.decoder_algorithm import deconvolution_decoder
from spikify.encoders.temporal.deconvolution import (
    bens_spiker,
    hough_spiker,
    modified_hough_spiker,
)


class TestDeconvolutionDecoderInput(unittest.TestCase):

    def test_empty_spike_train_raises(self):
        with self.assertRaises(ValueError):
            deconvolution_decoder(
                np.array([]),
                fir_bank=np.array([[0.25, 0.5, 0.25]]).T,
                shift=np.array([0.0]),
            )

    def test_output_is_ndarray(self):
        spikes = np.array([0, 1, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertIsInstance(result, np.ndarray)

    def test_1d_input_returns_2d_output(self):
        spikes = np.array([0, 1, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.ndim, 2)

    def test_2d_input_returns_2d_output(self):
        spikes = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]).T
        shift = np.array([0.0, 0.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.ndim, 2)

    def test_output_length_matches_input(self):
        spikes = np.array([0, 1, 0, 0, 1], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(len(result), len(spikes))

    def test_output_shape_matches_2d_input(self):
        spikes = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]).T
        shift = np.array([0.0, 0.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.shape, spikes.shape)

    def test_all_zeros_with_no_norm_returns_only_shift(self):
        spikes = np.array([0, 0, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.1])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=None)
        expected = np.full((4, 1), 0.1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_norm_none_applies_only_shift(self):
        spikes = np.array([0, 0, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.5])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=None)
        self.assertTrue(np.all(result == 0.5))

    def test_norm_provided_scales_output(self):
        spikes = np.array([1, 0, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.0])
        norm = np.array([2.0])
        result_with_norm = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        result_no_norm = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=None)
        # with norm=2.0, output should be 2x larger (before shift)
        np.testing.assert_array_almost_equal(result_with_norm, result_no_norm * 2.0)

    def test_single_spike_produces_nonzero_output(self):
        spikes = np.array([1, 0, 0, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift = np.array([0.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertTrue(np.any(result != 0.0))

    def test_shift_offsets_output(self):
        spikes = np.array([0, 0, 0, 0], dtype=np.int8)
        fir_bank = np.array([[0.25, 0.5, 0.25]]).T
        shift_a = np.array([0.0])
        shift_b = np.array([1.0])
        result_a = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift_a)
        result_b = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift_b)
        np.testing.assert_array_almost_equal(result_b - result_a, np.full((4, 1), 1.0))

    def test_multi_feature_output_shape(self):
        spikes = np.array([[0, 1], [1, 0], [0, 0], [0, 1]], dtype=np.int8)
        fir = np.array([0.25, 0.5, 0.25])
        fir_bank = np.stack([fir, fir], axis=0).T
        shift = np.array([0.0, 0.1])
        norm = np.array([1.0, 2.0])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape, spikes.shape)

    def test_multi_feature_independent_shift(self):
        spikes = np.zeros((4, 2), dtype=np.int8)
        fir = np.array([0.25, 0.5, 0.25])
        fir_bank = np.stack([fir, fir], axis=0).T
        shift = np.array([0.2, 0.8])
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=None)
        np.testing.assert_array_almost_equal(result[:, 0], np.full(4, 0.2))
        np.testing.assert_array_almost_equal(result[:, 1], np.full(4, 0.8))


class TestDeconvolutionDecoderRoundTripBSA(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([0.1, 0.2, 0.8, 0.95, 0.5, 0.3, 0.1]).reshape(-1, 1)
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([0.1, 0.2, 0.8, 0.95, 0.5, 0.3, 0.1])
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_flat_signal_no_spikes(self):
        signal = np.array([0.5, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        self.assertTrue(np.all(spikes == 0))
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        np.testing.assert_array_almost_equal(result, np.full_like(signal, shift[0]))

    def test_round_trip_positive_signal_no_shift(self):
        signal = np.array([0.1, 0.3, 0.6, 0.8, 0.5])
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        np.testing.assert_array_almost_equal(shift, np.array([0.0]))

    def test_round_trip_negative_signal_applies_shift(self):
        signal = np.array([-0.5, 0.1, 0.3, 0.6, 0.2])
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        self.assertAlmostEqual(shift[0], -0.5)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.shape[0], len(signal))

    def test_round_trip_bsa_does_not_return_norm(self):
        signal = np.array([0.1, 0.3, 0.6, 0.8, 0.5, 0.2, 0.4])
        result_tuple = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        # BSA returns (spikes, fir_bank, shift) — no norm
        self.assertEqual(len(result_tuple), 3)

    def test_round_trip_multi_feature(self):
        signal = np.array([[0.1, 0.5], [0.3, 0.4], [0.6, 0.9], [0.8, 0.7], [0.5, 0.3], [0.2, 0.1]])
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_high_threshold_no_spikes(self):
        signal = np.array([0.1, 0.3, 0.6, 0.8, 0.5, 0.2, 0.4]).reshape(-1, 1)
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=999.0)
        self.assertTrue(np.all(spikes == 0))
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        np.testing.assert_array_almost_equal(result, np.full_like(signal, shift[0]))

    def test_round_trip_output_non_negative_for_non_negative_signal(self):
        signal = np.array([0.1, 0.4, 0.7, 0.9, 0.6, 0.3, 0.2])
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.05)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertTrue(np.all(result >= 0))

    def test_round_trip_large_amplitude_signal(self):
        signal = np.array([1.0, 2.5, 5.0, 3.0, 1.5, 0.5, 0.2])
        spikes, fir_bank, shift = bens_spiker(signal, window_length=3, cutoff=0.1, threshold=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift)
        self.assertEqual(result.shape[0], len(signal))


class TestDeconvolutionDecoderRoundTripMHSA(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1]).reshape(-1, 1)
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_mhsa_returns_norm(self):
        signal = np.array([0.1, 0.3, 0.6, 0.8, 0.5, 0.2, 0.4])
        result_tuple = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        # MHSA returns (spikes, fir_bank, shift, norm)
        self.assertEqual(len(result_tuple), 4)

    def test_round_trip_negative_signal_applies_shift(self):
        signal = np.array([-0.5, 0.1, 0.3, 0.6, 0.2, 0.4, 0.1])
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        self.assertAlmostEqual(shift[0], -0.5)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape[0], len(signal))

    def test_round_trip_multi_feature(self):
        signal = np.array([[0.1, 0.5], [0.3, 0.4], [0.4, 0.6], [0.2, 0.3], [0.5, 0.7], [0.6, 0.8]])
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_large_amplitude_applies_norm(self):
        signal = np.array([1.0, 2.5, 5.0, 3.0, 1.5, 0.5, 0.2])
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        # norm should be > 1 since signal exceeds 1 after shift
        self.assertTrue(norm[0] > 1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape[0], len(signal))

    def test_round_trip_only_unipolar_spikes(self):
        signal = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 0.3, 0.1])
        spikes, fir_bank, shift, norm = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.5)
        self.assertTrue(np.all(spikes >= 0))


class TestDeconvolutionDecoderRoundTripHSA(unittest.TestCase):

    def test_round_trip_output_shape(self):
        signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1]).reshape(-1, 1)
        spikes, fir_bank, shift, norm = hough_spiker(signal, window_length=3, cutoff=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_output_is_ndarray(self):
        signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1])
        spikes, fir_bank, shift, norm = hough_spiker(signal, window_length=3, cutoff=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertIsInstance(result, np.ndarray)

    def test_round_trip_hsa_returns_norm(self):
        signal = np.array([0.1, 0.3, 0.6, 0.8, 0.5, 0.2, 0.4])
        result_tuple = hough_spiker(signal, window_length=3, cutoff=0.1)
        # HSA returns (spikes, fir_bank, shift, norm)
        self.assertEqual(len(result_tuple), 4)

    def test_round_trip_only_unipolar_spikes(self):
        signal = np.array([0.1, 0.2, 4.1, 1.0, 3.0, 0.3, 0.1])
        spikes, fir_bank, shift, norm = hough_spiker(signal, window_length=3, cutoff=0.1)
        self.assertTrue(np.all(spikes >= 0))

    def test_round_trip_negative_signal_applies_shift(self):
        signal = np.array([-0.5, 0.1, 0.3, 0.6, 0.2, 0.4, 0.1])
        spikes, fir_bank, shift, norm = hough_spiker(signal, window_length=3, cutoff=0.1)
        self.assertAlmostEqual(shift[0], -0.5)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape[0], len(signal))

    def test_round_trip_large_amplitude_applies_norm(self):
        signal = np.array([1.0, 2.5, 5.0, 3.0, 1.5, 0.5, 0.2])
        spikes, fir_bank, shift, norm = hough_spiker(signal, window_length=3, cutoff=0.1)
        self.assertTrue(norm[0] > 1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape[0], len(signal))

    def test_round_trip_multi_feature(self):
        signal = np.array([[0.1, 0.5], [0.3, 0.4], [0.4, 0.6], [0.2, 0.3], [0.5, 0.7], [0.6, 0.8]])
        spikes, fir_bank, shift, norm = hough_spiker(signal, window_length=3, cutoff=0.1)
        result = deconvolution_decoder(spikes, fir_bank=fir_bank, shift=shift, norm=norm)
        self.assertEqual(result.shape, signal.shape)

    def test_round_trip_hsa_vs_mhsa_same_signal_different_sparsity(self):
        # HSA is stricter than MHSA, so it should produce fewer or equal spikes
        signal = np.array([0.1, 0.2, 0.5, 0.9, 0.7, 0.4, 0.2])
        hsa_spikes, _, _, _ = hough_spiker(signal, window_length=3, cutoff=0.1)
        mhsa_spikes, _, _, _ = modified_hough_spiker(signal, window_length=3, cutoff=0.1, threshold=0.3)
        self.assertLessEqual(hsa_spikes.sum(), mhsa_spikes.sum())

    def test_round_trip_window_length_equals_signal_length_raises(self):
        signal = np.array([0.1, 0.3, 0.5])
        with self.assertRaises(ValueError):
            hough_spiker(signal, window_length=4, cutoff=0.1)
