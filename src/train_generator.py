"""
This script trains the generator.
--------
Requirements
--------
Before running this script make sure :
   - You have updated the DISCRIMINATOR_ID and GENERATOR_ID
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from networks.generator.generator import Generator, IMAGE_SIZE, NOISE_LAYERS
from networks.discriminator.discriminator import Discriminator


GENERATOR_ID = 5
DISCRIMINATOR_ID = 4

# ----- Initialisation of the parameters ----- #
# Batch size during training
BATCH_SIZE = 20

# Number of training epochs
NB_BATCHS = 5

# Learning rate for optimizers
LEARNING_RATE = 0.0002

# BETA_1 hyperparam for Adam optimizers
BETA_1 = 0.5

# Path to register the generated images
PATH_FAKE_DATA = Path("../SRTM_data/Fake")

# Clean the folder where fake images are registered #
# erase_folder(PATH_FAKE_DATA)


# --- Creation of the networks --- #
NET_D = Discriminator(IMAGE_SIZE)
NET_G = Generator(0)


# --- Loading the optimized disciminator --- #
PATH_TRAIN_D = Path("../networks_data/discriminator")
NET_D.load_state_dict(torch.load(
    PATH_TRAIN_D / (str(DISCRIMINATOR_ID) + ".pth")
))

# --- Loading the optimized generator --- #
PATH_TRAIN_G = Path("../networks_data/generator")
NET_G.load_state_dict(torch.load(
    PATH_TRAIN_G / (str(GENERATOR_ID - 1) + ".pth")
))

# --- Definition of the optimizer (the one who computes the gradient) --- #
OPTIMIZER_G = optim.Adam(NET_G.parameters(), LEARNING_RATE, betas=(BETA_1, 0.999))


# --- Definition of the CRITERION --- #
CRITERION = nn.CrossEntropyLoss()


print("Starting Training Loop...")
# For each epoch
for epoch in range(NB_BATCHS):
    print("training epoch:", epoch)

    # --- Reset of the G --- #
    NET_G.zero_grad()

    # Generate batch of latent vectors
    noise = torch.randn(BATCH_SIZE, NOISE_LAYERS, 1, 1)
    conditions = torch.randn(BATCH_SIZE, 2)

    # --- Initialisation of the labels ---#
    # Generate fake image batch with G
    fake = NET_G(noise)

    label = torch.zeros(BATCH_SIZE, dtype=torch.long)

    # --- Reactions of D --- #
    # Test D with the generated images
    output = NET_D(fake)

    with torch.no_grad():
        display = NET_D.soft_max(NET_D(fake))

        n = 0
        avg = 0
        for probas in display:
            avg += probas[0]
            n += 1

        avg /= n

        # Displays the precision
        print("The probability of misleading the discriminator is {}".format(avg))

    # Calculate G's loss based on this output
    errG = CRITERION(output, label)
    # Calculate gradients for G
    errG.backward()

    # --- Update G --- #
    OPTIMIZER_G.step()

torch.save(NET_G.state_dict(), PATH_TRAIN_G / (str(GENERATOR_ID) + ".pth"))
