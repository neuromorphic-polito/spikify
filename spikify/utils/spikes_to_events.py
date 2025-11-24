import numpy as np


def spikes_to_events(encoded_signal: np.ndarray, fs: float) -> np.ndarray:

    match encoded_signal.ndim:
        case 1:
            encoded_signal = encoded_signal[np.newaxis, :, np.newaxis]
        case 2:
            encoded_signal = encoded_signal[np.newaxis, :, :]

    time_array = np.arange(encoded_signal.shape[1]) * int(1 / fs * 1_000_000)  # in microseconds
    # time_array = np.reshape(time_array, (1, -1))  # (1, time_indices)

    spike_positions = np.where(encoded_signal != 0)  # (sensor_ids, time_indices, channel_ids)

    event_times = time_array[spike_positions[1]]
    sensor_ids = spike_positions[0]
    channel_ids = spike_positions[2]
    spike_values = encoded_signal[spike_positions]
    polarities = spike_values > 0

    events = np.zeros(len(event_times), dtype=np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), (("p", "on"), "?")]))

    events["t"] = event_times.astype(np.uint64)
    events["x"] = sensor_ids.astype(np.uint16)
    events["y"] = channel_ids.astype(np.uint16)
    events["on"] = polarities.astype(bool)

    return events
