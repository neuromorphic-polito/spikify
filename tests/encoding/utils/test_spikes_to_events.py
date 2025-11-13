import unittest
import numpy as np
from spikify.utils import spikes_to_events


class TestSpikesToEvents(unittest.TestCase):
    """Tests for the spikes_to_events function."""

    def test_1d_input(self):
        """Test with 1D input array."""
        encoded_signal = np.array([0, 1, 0, -1, 0])
        fs = 1000  # 1 kHz
        events = spikes_to_events(encoded_signal, fs)

        expected_times = np.array([1000, 3000], dtype=np.uint64)
        expected_x = np.array([0, 0], dtype=np.uint16)
        expected_y = np.array([0, 0], dtype=np.uint16)
        expected_on = np.array([True, False], dtype=bool)

        np.testing.assert_array_equal(events["t"], expected_times)
        np.testing.assert_array_equal(events["x"], expected_x)
        np.testing.assert_array_equal(events["y"], expected_y)
        np.testing.assert_array_equal(events["on"], expected_on)
