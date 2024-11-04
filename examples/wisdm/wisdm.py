from pathlib import Path
import pandas as pd
import numpy as np
from spikify.encoding.rate import poisson_rate
from spikify.encoding.temporal.latency import burst_encoding
from spikify.encoding.temporal.contrast import (
    moving_window,
    step_forward,
    threshold_based_representation,
    zero_cross_step_forward,
)
from spikify.encoding.temporal.global_referenced import phase_encoding, time_to_first_spike
from spikify.encoding.temporal.deconvolution import ben_spiker, hough_spiker, modified_hough_spiker

activities = [
    (0, "A", "walking"),
    (1, "B", "jogging"),
    (2, "C", "stairs"),
    (3, "D", "sitting"),
    (4, "E", "standing"),
    (5, "F", "typing"),
    (6, "G", "brushing_teeth"),
    (7, "H", "eating_soup"),
    (8, "I", "eating_chips"),
    (9, "J", "eating_pasta"),
    (10, "K", "drinking"),
    (11, "L", "eating_sandwich"),
    (12, "M", "kicking_soccer"),
    (13, "O", "catch_tennis"),
    (14, "P", "dribbling_basketball"),
    (15, "Q", "writing"),
    (16, "R", "clapping"),
    (17, "S", "folding_clothes"),
]


# Helper function to apply encoding
def apply_encoding(
    dataframe,
    path,
    device,
    sensor_type,
    subject,
    encoding_type,
    encoding_fn,
    **encoding_args,
):
    dataframe["x"] = dataframe.groupby("label_id")["x"].transform(lambda x: encoding_fn(np.array(x), **encoding_args))
    dataframe["y"] = dataframe.groupby("label_id")["y"].transform(lambda x: encoding_fn(np.array(x), **encoding_args))
    dataframe["z"] = dataframe.groupby("label_id")["z"].transform(lambda x: encoding_fn(np.array(x), **encoding_args))
    save_encoded_data(dataframe, path, device, sensor_type, subject, encoding_type)
    return dataframe


# Helper function to save encoded data
def save_encoded_data(dataframe, path, device, sensor_type, subject, encoding_type):
    dir_path = path / encoding_type / device / sensor_type
    dir_path.mkdir(parents=True, exist_ok=True)
    # Check if file exists
    file_path = dir_path / f"data_{subject + 1600}_{sensor_type}_{device}_spike.csv"
    write_header = not file_path.exists()  # Write header if file doesn't exist

    dataframe.to_csv(
        dir_path / f"data_{subject + 1600}_{sensor_type}_{device}_spike.csv", index=False, mode="a", header=write_header
    )


def convert_dataset_into_spike(path):

    # min_len = 3500
    devices = ["phone", "watch"]
    subjects = 51
    labels = [activity[1] for activity in activities]

    for device in devices:
        for subject in range(subjects):
            # Read accelerometer and gyroscope data
            dataframe_accel = pd.read_csv(path / device / "accel" / f"data_{subject + 1600}_accel_{device}.csv")
            dataframe_gyro = pd.read_csv(path / device / "gyro" / f"data_{subject + 1600}_gyro_{device}.csv")

            # Filter by label_ids present in the labels list
            dataframe_accel = dataframe_accel[dataframe_accel["label_id"].isin(labels)]
            dataframe_gyro = dataframe_gyro[dataframe_gyro["label_id"].isin(labels)]

            # Limit samples per label
            dataframe_accel = (
                dataframe_accel.assign(count=dataframe_accel.groupby("label_id").cumcount())
                .query("count < @min_len")
                .drop(columns="count")
            )

            dataframe_gyro = (
                dataframe_gyro.assign(count=dataframe_gyro.groupby("label_id").cumcount())
                .query("count < @min_len")
                .drop(columns="count")
            )

            # Apply Poisson rate encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "poisson_encoded",
                poisson_rate,
                interval_length=20,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "poisson_encoded",
                poisson_rate,
                interval_length=20,
            )
            #
            #
            # Apply Burst encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "burst_encoded",
                burst_encoding,
                n_max=5,
                t_min=0,
                t_max=4,
                length=14,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "burst_encoded",
                burst_encoding,
                n_max=5,
                t_min=0,
                t_max=4,
                length=14,
            )
            #
            # Apply Moving Window encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "moving_window_encoded",
                moving_window,
                window_length=3,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "moving_window_encoded",
                moving_window,
                window_length=3,
            )
            #
            # Apply Threshold Based Representation encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "threshold_based_encoded",
                threshold_based_representation,
                factor=0.5,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "threshold_based_encoded",
                threshold_based_representation,
                factor=0.5,
            )
            #
            # Apply Phase Encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "phase_encoded",
                phase_encoding,
                num_bits=5,
            )
            apply_encoding(
                dataframe_gyro.copy(), path.parent, device, "gyro", subject, "phase_encoded", phase_encoding, num_bits=5
            )
            #
            # Apply Time to First Spike encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "time_to_first_spike_encoded",
                time_to_first_spike,
                interval=10,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "time_to_first_spike_encoded",
                time_to_first_spike,
                interval=10,
            )
            #
            # Apply Ben Spiker encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "ben_spiker_encoded",
                ben_spiker,
                window_length=3,
                threshold=1.0,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "ben_spiker_encoded",
                ben_spiker,
                window_length=3,
                threshold=1.0,
            )
            #
            # Apply Hough Spiker encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "hough_spiker_encoded",
                hough_spiker,
                window_length=3,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "hough_spiker_encoded",
                hough_spiker,
                window_length=3,
            )
            #
            # Apply Modified Hough Spiker encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path.parent,
                device,
                "accel",
                subject,
                "modified_hough_spiker_encoded",
                modified_hough_spiker,
                window_length=3,
                threshold=0.85,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path.parent,
                device,
                "gyro",
                subject,
                "modified_hough_spiker_encoded",
                modified_hough_spiker,
                window_length=3,
                threshold=0.85,
            )

            for label in labels:
                df_accel_label = dataframe_accel[dataframe_accel["label_id"] == label]
                df_gyro_label = dataframe_gyro[dataframe_gyro["label_id"] == label]

                # Stack accelerometer and gyroscope data for the current label
                stacked_data = np.vstack(
                    [df_accel_label[["x", "y", "z"]].values, df_gyro_label[["x", "y", "z"]].values]
                )

                # Calculate sfThreshold and zcsfThreshold for the current label
                sfThreshold = np.mean([axis.max() - axis.min() for axis in stacked_data]) / 10
                zcsfThreshold = np.mean([axis.max() - axis.min() for axis in stacked_data]) / 10

                if np.isnan(sfThreshold) or np.isnan(zcsfThreshold):
                    continue

                # Apply Step Forward encoding
                apply_encoding(
                    df_accel_label.copy(),
                    path.parent,
                    device,
                    "accel",
                    subject,
                    "step_forward_encoded",
                    step_forward,
                    threshold=sfThreshold,
                )
                apply_encoding(
                    df_gyro_label.copy(),
                    path.parent,
                    device,
                    "gyro",
                    subject,
                    "step_forward_encoded",
                    step_forward,
                    threshold=sfThreshold,
                )

                # Apply Zero Cross Step Forward encoding
                apply_encoding(
                    df_accel_label.copy(),
                    path.parent,
                    device,
                    "accel",
                    subject,
                    "zero_cross_step_forward_encoded",
                    zero_cross_step_forward,
                    threshold=zcsfThreshold,
                )
                apply_encoding(
                    df_gyro_label.copy(),
                    path.parent,
                    device,
                    "gyro",
                    subject,
                    "zero_cross_step_forward_encoded",
                    zero_cross_step_forward,
                    threshold=zcsfThreshold,
                )


if __name__ == "__main__":
    path = Path("../../datasets/wisdm-dataset/raw")
    convert_dataset_into_spike(path)
