"""
This scrip construct the data base composed of real images,
and saves it to SRTM_data/Real.
--------
Requirements
--------
Before running this script make sure :
   - You have downloaded the zip containing the real images,
   and saved them in ~/SRTM_zip
"""

from pathlib import Path
import os
import random
from pre_processing.unzip import unzip
from pre_processing.renormalizing import open_tif, normalize, cut


# -- Paths -- #
ZIP_ORIGINAL_PATH = Path("../SRTM_zip")
DESTINATION_PATH_REAL_IMAGES = Path("../SRTM_data/Real")
DESTINATION_PATH_REAL_IMAGES.mkdir(parents=True, exist_ok=True)

# Remove previously generated images
print("Emptying the destination folder ...")
REAL_FILES = os.listdir(DESTINATION_PATH_REAL_IMAGES)
for image in REAL_FILES:
    os.remove(DESTINATION_PATH_REAL_IMAGES / image)

# -- Parameters -- #
# Dimension of sample data
WANTED_DIM_IMAGES = 100
# Size of sample data
NB_TIF_TAKEN = 3


# -- We unzip the zipped images -- #
print("Unzipping...")
ZIPS = os.listdir(ZIP_ORIGINAL_PATH)
random.shuffle(ZIPS)
ZIPS = ZIPS[:NB_TIF_TAKEN]

unzip(ZIP_ORIGINAL_PATH, ZIPS, DESTINATION_PATH_REAL_IMAGES)


# -- We take the .tif. We cut them and save the interesting ones -- #
TIFS = os.listdir(DESTINATION_PATH_REAL_IMAGES)
# For each big image, we take the interesting part
print("generating real images from", NB_TIF_TAKEN, "tifs")
for name in TIFS:
    print("Cutting", name)
    file_path = DESTINATION_PATH_REAL_IMAGES / name
    if name[-4:] == '.tif':
        image = open_tif(file_path)
        image_norm = normalize(image)
        cut(image_norm, DESTINATION_PATH_REAL_IMAGES, name[: -4], WANTED_DIM_IMAGES)
    # Tiff has been cut, it is no longer needed
    os.remove(file_path)
