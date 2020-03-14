"""
This script allow you to see how well the training of
the generator went.
--------
You may use this script in the end of the training.
--------
Requirements
--------
Before running this script make sure :
   - You have trained the generator and the discriminator
   - You have updated the TRAINING_ID and GENERATOR_ID
"""

from pathlib import Path
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch import load, device, randn, LongTensor, softmax, tanh
from torch import nn
from torch.cuda import is_available
from networks.discriminator.discriminator import Discriminator
from networks.generator.generator import Generator, NB_GPU, NOISE_LAYERS, IMAGE_SIZE
from utils.image3d import show_3d


# --- Parameters --- #
# Decide which device we want to run on
AVAILABLE = is_available()
DEVICE = device("cuda:0" if (AVAILABLE and NB_GPU > 0) else "cpu")

# Number of images generated
NB_GENERATION = 1

# DISCRIMINATOR_ID is the number of the next training for the discriminator
# GENERATOR_ID is the number of the next training for the generator
DISCRIMINATOR_ID = 4
GENERATOR_ID = 4


# --- Creation of the discriminator and the generator --- #
DISCRIMINATOR = Discriminator(IMAGE_SIZE)
GENERATOR = Generator(NB_GPU).to(DEVICE)
# Optimisation of the discriminator
if (DEVICE.type == 'cuda') and (NB_GPU > 1):
    GENERATOR = nn.DataParallel(GENERATOR, list(range(NB_GPU)))


# --- Loading the optimized discriminator and the optimized generator --- #
# Paths to the networks previous training data
DISCRIMINATOR_DATA = Path("../networks_data/discriminator")
GENERATOR_DATA = Path("../networks_data/generator")

# instances of the networks
DISCRIMINATOR.load_state_dict(load(
    DISCRIMINATOR_DATA / (str(DISCRIMINATOR_ID) + ".pth")
))

GENERATOR.load_state_dict(load(
    GENERATOR_DATA / (str(GENERATOR_ID) + ".pth")
))


# --- Initialisation of the generator --- #
NOISE = randn(NB_GENERATION, NOISE_LAYERS, 1, 1, device=DEVICE)

CONDITIONS = randn(NB_GENERATION, 2)

SWAP = LongTensor([1, 0])
if CONDITIONS[0][0] < CONDITIONS[0][1]:
    CONDITIONS[0] = CONDITIONS[0][SWAP]
CONDITIONS = tanh(CONDITIONS).detach()


# --- Generation of the images --- #
FAKE = GENERATOR(NOISE, CONDITIONS).detach()


# --- Print the score of the fake images with the discriminator --- #
SOFTMAX = nn.Softmax(1)
OUTPUT = softmax(DISCRIMINATOR(FAKE), 1)
PROBA = OUTPUT[:, 0].mean()
print("La probabilité de générer une bonne carte est {}".format(PROBA))
print("Conditions demandées: " + str(CONDITIONS))
print("Conditions obtenues: " + str(FAKE.max()) + " " + str(FAKE.mean()))

# --- Plots the image obtained --- #
IMAGE = vutils.make_grid(FAKE, padding=2, normalize=True)
plt.imshow(np.transpose(IMAGE, (1, 2, 0)))
show_3d(FAKE[0, 0].detach().numpy())
plt.show()
