"""
This scrip construct the data base composed of fake images
created with fractal noise, and saves it to ~/SRTM_data/Fake.
--------
It creates as much images as in real images.
--------
Requirements
--------
Before running this script make sure :
   - You have created a file with real images (use real_images.py).
"""

from pathlib import Path
import os
import numpy as np
from utils.noise.fractal_noise import FractalNoise


# -- Paths -- #
REAL_DATA_PATH = Path("../SRTM_data/Real")
DESTINATION_PATH_FAKE_IMAGES = Path("../SRTM_data/Fake")
DESTINATION_PATH_FAKE_IMAGES.mkdir(parents=True, exist_ok=True)

# Remove previously generated images
print("emptying destination folder")
FAKE_FILES = os.listdir(DESTINATION_PATH_FAKE_IMAGES)
for image in FAKE_FILES:
    os.remove(DESTINATION_PATH_FAKE_IMAGES / image)

# -- Parameters -- #
# Dimension of sample data
WANTED_DIM_IMAGES = 100
# Number of images to generate (depends on number of real images)
WANTED_NB_IMAGES = len(os.listdir(REAL_DATA_PATH))

print("generating", WANTED_NB_IMAGES, "images")
for i in range(WANTED_NB_IMAGES):
    if i != 0 and i % 100 == 0:
        print(
            i, "images generated,",
            str(int(i / WANTED_NB_IMAGES * 100)) + "%"
        )

    noise = FractalNoise(5, 0.3)
    img = noise.draw(WANTED_DIM_IMAGES, WANTED_DIM_IMAGES, x_incr=0.006, y_incr=0.006)
    destination_path = DESTINATION_PATH_FAKE_IMAGES / str(i)
    np.save(destination_path, img)
