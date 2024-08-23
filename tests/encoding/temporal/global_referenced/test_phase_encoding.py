from spikify.encoding.temporal.global_referenced.phase_encoding_algorithm import phase_encoding
import unittest
import numpy as np


class TestPhaseEncodingAlgorithm(unittest.TestCase):
    """Tests phase_encoding function."""

    def test_basic_functionality(self):
        """Ensure the function correctly generates spikes when the signal contains patterns that match the filter
        window and threshold conditions."""

        signal = np.array([0, 1.5, 2, 3, 4, 5, 6, 3, 2, 1])
        num_bits = 2
        expected_spikes = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1])
        result = phase_encoding(signal, num_bits)
        np.testing.assert_array_equal(result, expected_spikes)
