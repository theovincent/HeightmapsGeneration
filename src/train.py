""" This scrip trains the discriminator and the generator
in the same time.
--------
You may use this script after using train_discriminator.
You should use train_discriminator.py before because in
this script it is better if the discriminator has already
been trained.
--------
Requirements
--------
Before running this script make sure :
   - You have downloaded the data base in zip files
   - You have updated the TRAINING_ID and GENERATOR_ID
--------
If "srtm_A_B is not interesting" is printed to often,
please lower the BATCH_SIZE;
"""

from pathlib import Path
from torch import zeros, long, ones, randn, LongTensor, tanh, load, no_grad, save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from numpy import sqrt
from utils.verif import loss
from pre_processing.renormalizing import create_batch
from networks.discriminator.discriminator import Discriminator
from networks.generator.generator import Generator, NOISE_LAYERS

# DISCRIMINATOR_ID is the number of the next training for the discriminator
# GENERATOR_ID is the number of the next training for the generator
DISCRIMINATOR_ID = 5
GENERATOR_ID = 5

# Paths to the networks previous training data
DISCRIMINATOR_DATA = Path("../networks_data/discriminator")
GENERATOR_DATA = Path("../networks_data/generator")

# Path to real zips and images (temporary) (for discriminator training)
SRTM_ZIP = Path("../SRTM_zip")
SRTM_TIF = Path("../SRTM_data/Real")

# instances of the networks
DISCRIMINATOR = Discriminator(100)
DISCRIMINATOR.load_state_dict(load(
    DISCRIMINATOR_DATA / (str(DISCRIMINATOR_ID - 1) + ".pth")
))

GENERATOR = Generator(0)
GENERATOR.load_state_dict(load(
    GENERATOR_DATA / (str(GENERATOR_ID - 1) + ".pth")
))

# Training parameters
BATCH_SIZE = 10
NB_BATCH = 50

IMAGE_SIZE = 100

# Labels
# Real images => label = 0
REAL_LABEL_D = zeros(BATCH_SIZE, dtype=long)
# Discriminator wants fake_images to be detected => label = 1
FAKE_LABEL_D = ones(BATCH_SIZE, dtype=long)
# Generator wants fake images to look real => label = 0
FAKE_LABEL_G = zeros(BATCH_SIZE, dtype=long)

# Loss function
CRITERION = CrossEntropyLoss()

# optimizers
OPTIM_G = Adam(GENERATOR.parameters(), lr=0.0002, betas=(0.5, 0.99))
OPTIM_D = Adam(DISCRIMINATOR.parameters(), lr=0.0002, betas=(0.5, 0.99))


for i in range(NB_BATCH):
    ###########################
    # -- Initialisation of the networks -- #
    DISCRIMINATOR.zero_grad()
    GENERATOR.zero_grad()

    # -- Loading of the batches -- #
    # Load real batch
    real_batch = create_batch(SRTM_ZIP, SRTM_TIF, BATCH_SIZE, IMAGE_SIZE)
    # Load fake batch

    noise = randn(BATCH_SIZE, NOISE_LAYERS, 1, 1)

    # Load conditions
    conditions = randn(BATCH_SIZE, 2)

    # If max < mean we swap the two values
    swap = LongTensor([1, 0])
    for j in range(BATCH_SIZE):
        if conditions[j][0] < conditions[j][1]:
            conditions[j] = conditions[j][swap]
    conditions = tanh(conditions).detach()

    fake_batch = GENERATOR(noise, conditions)

    # -- Run into the discriminator -- #
    output_realD = DISCRIMINATOR(real_batch)  # For the training of D
    output_fakeD = DISCRIMINATOR(fake_batch.detach())  # For the training of D
    # The fake_batch's grad has to be computed because it uses the generator
    output_fakeG = DISCRIMINATOR(fake_batch)  # For the training of G

    # -- Computation of the errors -- #
    errD_real = CRITERION(output_realD, REAL_LABEL_D)
    errD_fake = CRITERION(output_fakeD, FAKE_LABEL_D)
    errD = errD_real + errD_fake

    errG_img = CRITERION(output_fakeG, FAKE_LABEL_G)
    errG_cond = loss(fake_batch, conditions)
    errG = errG_img + errG_cond

    err = errD + errG
    err.backward()

    # -- Updates networks -- #
    OPTIM_D.step()
    OPTIM_G.step()

    ###########################
    # Results display:
    with no_grad():
        real_avg = DISCRIMINATOR.soft_max(output_realD)[:, 0].mean()
        fake_avg = DISCRIMINATOR.soft_max(output_fakeD)[:, 0].mean()

        print("batch " + str(i) + ":")
        print("average proba of real: " + str(real_avg))
        print("average proba of fake: " + str(fake_avg))
        print(
            "average distance with conditions: "
            + str(errG_cond / (2 * sqrt(BATCH_SIZE))) + "\n"
        )


# Saves training state of both networks
save(
    DISCRIMINATOR.state_dict(),
    DISCRIMINATOR_DATA / (str(DISCRIMINATOR_ID) + ".pth")
)
save(
    GENERATOR.state_dict(),
    GENERATOR_DATA / (str(GENERATOR_ID) + ".pth")
)
