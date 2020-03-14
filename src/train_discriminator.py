""" This script trains the discriminator.
--------
You may use this script in the beginning of the training.
You should use train.py after because the script is only useful
to start the training since the fake images are generated by
fractal noise and not by the generator.
--------
Requirements
--------
Before running this script make sure :
   - You have downloaded real images and unzipped real images (real_images.py).
   - You have created fake images with fake_images_fractal.py.
   - You have updated the TRAINING_ID.
"""

from pathlib import Path
import numpy as np
from torch import from_numpy, float32, long, load, save
from networks.discriminator.discriminator import Discriminator
from networks.discriminator.shuffle_data_set_discriminator import shuffle_data_set_d

# TRAINING_ID is the number of the next training
TRAINING_ID = 5

# STATE_PATH is the path where the last training of the discriminator lies
STATE_PATH = Path("../networks_data/discriminator/")

# Instance of the generator network we want to train
DISCRIMINATOR = Discriminator(100)

# Loads state from previous training (O.pth is randomly generated)
print("loading discriminator state")
DISCRIMINATOR.load_state_dict(load(
    STATE_PATH / (str(TRAINING_ID - 1) + ".pth")
))

SIZE_BATCH = 10
NB_EPOCH = 2

# Imports and shuffles real and fake training images
print("loading data set")
(DATA_SET, LABELS) = shuffle_data_set_d(
    Path("../SRTM_data/Real"),
    Path("../SRTM_data/Fake")
)
NB_IMAGES = len(DATA_SET)

# Creates tensors with loaded images and their labels
DATA_SET = np.array(DATA_SET)
LABELS = np.array(LABELS)

DATA_SET = from_numpy(DATA_SET)
LABELS = from_numpy(LABELS)

# Reorganizes them in batches
DIM = DATA_SET.shape

NB_IMAGES = (NB_IMAGES // SIZE_BATCH) * SIZE_BATCH
print(NB_IMAGES)

DATA_SET = DATA_SET[: NB_IMAGES].view(-1, SIZE_BATCH, 1, DIM[-2], DIM[-1])
LABELS = LABELS[:NB_IMAGES].view(-1, SIZE_BATCH)

# Casts data type for torch
DATA_SET = DATA_SET.to(float32)
LABELS = LABELS.to(long)

print("Starting training:")
for epoch in range(NB_EPOCH):
    print("training epoch:", epoch, "\n")
    DISCRIMINATOR.custom_training(DATA_SET, LABELS)
    print("\n")

save(DISCRIMINATOR.state_dict(), STATE_PATH / (str(TRAINING_ID) + ".pth"))