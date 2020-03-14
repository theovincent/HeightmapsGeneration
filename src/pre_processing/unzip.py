"""
Defines initial function to extract the .tif contained in .zip archives

Functions:
    unzip(zip_location, zip_images, destination): unzips initial tif in destination folder
"""

from zipfile import ZipFile
import os
from pathlib import Path


def unzip(zip_location, zip_images, destination):
    """
    Unzip all the zips of zip_images located in zip_location

    Parameters:
        zip_location (pathlib.Path): directory containing the zips we want to extract

        zip_images (String): name of the zips inside zip_location to extract

        destination (pathlib.Path): directory to extract desired zips into
    """
    for name in zip_images:
        file_path = zip_location / name
        if name[-4:] == '.zip':
            # Create initial ZipFile Object and load sample.zip in it
            with ZipFile(file_path, 'r') as zip_obj:
                # Extract all the contents of zip file in different directory
                image_name = name[: -3] + 'tif'
                zip_obj.extract(image_name, destination)


if __name__ == "__main__":
    ZIP_PATH = Path("../../SRTM_zip")
    FILES = os.listdir(ZIP_PATH)
    DESTINATION_PATH = Path("../../SRTM_data/Test")
    unzip(ZIP_PATH, [FILES[0]], DESTINATION_PATH)
