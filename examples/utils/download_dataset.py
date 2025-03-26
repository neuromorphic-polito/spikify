import requests
from tqdm import tqdm
import zipfile
from pathlib import Path
from typing import Union
import os


def extract_zip_recursive(zip_path: Union[str, Path], extract_to: Union[str, Path]):
    """
    Recursively extracts zip files including any nested zip files found within. Each extracted zip file is then also
    opened and its contents extracted recursively into a new directory based on the zip file's name. Nested zip files
    are removed after extraction.

    Args:
        zip_path (Union[str, Path]): The path to the zip file that needs to be extracted.
                                     Can be a string or a Path object.
        extract_to (Union[str, Path]): The directory where the zip files should be extracted to.
                                       Can be a string or a Path object.

    """

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_files = zip_ref.infolist()
        with tqdm(total=len(zip_files), unit="file", desc=f"Extracting {zip_path}") as progress_bar:
            for file in zip_files:
                zip_ref.extract(file, path=extract_to)
                progress_bar.update(1)
                extracted_file_path = os.path.join(extract_to, file.filename)

                if zipfile.is_zipfile(extracted_file_path):
                    nested_extract_to = os.path.join(extract_to, os.path.splitext(file.filename)[0])
                    os.makedirs(nested_extract_to, exist_ok=True)
                    extract_zip_recursive(extracted_file_path, nested_extract_to)
                    os.remove(extracted_file_path)


def download_dataset(url, destination):

    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file. Error: {e}")
        return

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to download file. Status code: {response.status_code}")
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    # Get the total file size from the headers
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte

    try:

        # Initialize tqdm progress bar
        with open(destination, "wb") as file, tqdm(
            desc=destination,
            total=total_size,
            unit="iB",  # Unit of measure
            unit_scale=True,
            unit_divisor=1024,  # Convert to kilobytes
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        print(f"File successfully downloaded to {destination}")

    except Exception as error:
        print(f"Failed to download file. Error Message: {error}")
