from pathlib import Path
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
from spikify.encoding.temporal.deconvolution import bens_spiker, hough_spiker, modified_hough_spiker
from datasets import load_dataset

REPO_ID = "neuromorphic-polito/siddha"
data = load_dataset("neuromorphic-polito/siddha", "main_data", split="full")


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
    dataframe[f"{sensor_type}_x"] = dataframe.groupby("activity")[f"{sensor_type}_x"].transform(
        lambda x: encoding_fn(np.array(x), **encoding_args)
    )
    dataframe[f"{sensor_type}_y"] = dataframe.groupby("activity")[f"{sensor_type}_y"].transform(
        lambda x: encoding_fn(np.array(x), **encoding_args)
    )
    dataframe[f"{sensor_type}_z"] = dataframe.groupby("activity")[f"{sensor_type}_z"].transform(
        lambda x: encoding_fn(np.array(x), **encoding_args)
    )
    save_encoded_data(dataframe, path, device, sensor_type, subject, encoding_type)
    return dataframe


# Helper function to save encoded data
def save_encoded_data(dataframe, path, device, sensor_type, subject, encoding_type):
    dir_path = path / encoding_type / device / sensor_type
    dir_path.mkdir(parents=True, exist_ok=True)
    # Check if file exists
    file_path = dir_path / f"data_{subject}_{sensor_type}_{device}_spike.csv"
    write_header = not file_path.exists()  # Write header if file doesn't exist

    dataframe.to_csv(
        dir_path / f"data_{subject}_{sensor_type}_{device}_spike.csv", index=False, mode="a", header=write_header
    )


def convert_dataset_into_spike(dataframe):

    path = Path("./encoded_data")
    path.mkdir(parents=True, exist_ok=True)

    min_len = 3500
    devices = ["phone", "watch"]
    subjects = 51
    labels = [activity[1] for activity in activities]

    for device in devices:
        for subject in range(subjects):
            # Read accelerometer and gyroscope data
            dataframe_accel = dataframe[(dataframe["id"] == str(subject)) & (dataframe["device"] == device)].loc[
                :, ["acc_x", "acc_y", "acc_z", "activity", "timestamp"]
            ]
            dataframe_gyro = dataframe[(dataframe["id"] == str(subject)) & (dataframe["device"] == device)].loc[
                :, ["gyro_x", "gyro_y", "gyro_z", "activity", "timestamp"]
            ]

            # Filter by label_ids present in the labels list
            dataframe_accel = dataframe_accel[dataframe_accel["activity"].isin(labels)]
            dataframe_gyro = dataframe_gyro[dataframe_gyro["activity"].isin(labels)]

            # Limit samples per label
            dataframe_accel = (
                dataframe_accel.assign(count=dataframe_accel.groupby("activity").cumcount())
                .query(f"count < {min_len}")
                .drop(columns="count")
            )

            dataframe_gyro = (
                dataframe_gyro.assign(count=dataframe_gyro.groupby("activity").cumcount())
                .query(f"count < {min_len}")
                .drop(columns="count")
            )

            # Apply Poisson rate encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path,
                device,
                "acc",
                subject,
                "poisson_encoded",
                poisson_rate,
                interval_length=20,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
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
                path,
                device,
                "acc",
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
                path,
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
                path,
                device,
                "acc",
                subject,
                "moving_window_encoded",
                moving_window,
                window_length=3,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
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
                path,
                device,
                "acc",
                subject,
                "threshold_based_encoded",
                threshold_based_representation,
                factor=0.5,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
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
                path,
                device,
                "acc",
                subject,
                "phase_encoded",
                phase_encoding,
                num_bits=5,
            )
            apply_encoding(
                dataframe_gyro.copy(), path, device, "gyro", subject, "phase_encoded", phase_encoding, num_bits=5
            )
            #
            # Apply Time to First Spike encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path,
                device,
                "acc",
                subject,
                "time_to_first_spike_encoded",
                time_to_first_spike,
                interval=10,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
                device,
                "gyro",
                subject,
                "time_to_first_spike_encoded",
                time_to_first_spike,
                interval=10,
            )
            #
            # Apply Bens Spiker encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path,
                device,
                "acc",
                subject,
                "bens_spiker_encoded",
                bens_spiker,
                window_length=3,
                threshold=1.0,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
                device,
                "gyro",
                subject,
                "bens_spiker_encoded",
                bens_spiker,
                window_length=3,
                threshold=1.0,
            )
            #
            # Apply Hough Spiker encoding and save the data
            apply_encoding(
                dataframe_accel.copy(),
                path,
                device,
                "acc",
                subject,
                "hough_spiker_encoded",
                hough_spiker,
                window_length=3,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
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
                path,
                device,
                "acc",
                subject,
                "modified_hough_spiker_encoded",
                modified_hough_spiker,
                window_length=3,
                threshold=0.85,
            )
            apply_encoding(
                dataframe_gyro.copy(),
                path,
                device,
                "gyro",
                subject,
                "modified_hough_spiker_encoded",
                modified_hough_spiker,
                window_length=3,
                threshold=0.85,
            )

            for label in labels:
                df_accel_label = dataframe_accel[dataframe_accel["activity"] == label]
                df_gyro_label = dataframe_gyro[dataframe_gyro["activity"] == label]

                # Stack accelerometer and gyroscope data for the current label
                stacked_data = np.vstack(
                    [
                        df_accel_label[["acc_x", "acc_y", "acc_z"]].values,
                        df_gyro_label[["gyro_x", "gyro_y", "gyro_z"]].values,
                    ]
                )

                # Calculate sfThreshold and zcsfThreshold for the current label
                sfThreshold = np.mean([axis.max() - axis.min() for axis in stacked_data]) / 10
                zcsfThreshold = np.mean([axis.max() - axis.min() for axis in stacked_data]) / 10

                if np.isnan(sfThreshold) or np.isnan(zcsfThreshold):
                    continue

                # Apply Step Forward encoding
                apply_encoding(
                    df_accel_label.copy(),
                    path,
                    device,
                    "acc",
                    subject,
                    "step_forward_encoded",
                    step_forward,
                    threshold=sfThreshold,
                )
                apply_encoding(
                    df_gyro_label.copy(),
                    path,
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
                    path,
                    device,
                    "acc",
                    subject,
                    "zero_cross_step_forward_encoded",
                    zero_cross_step_forward,
                    threshold=zcsfThreshold,
                )
                apply_encoding(
                    df_gyro_label.copy(),
                    path,
                    device,
                    "gyro",
                    subject,
                    "zero_cross_step_forward_encoded",
                    zero_cross_step_forward,
                    threshold=zcsfThreshold,
                )


if __name__ == "__main__":
    wisdm_data = data.to_pandas()
    convert_dataset_into_spike(wisdm_data)
