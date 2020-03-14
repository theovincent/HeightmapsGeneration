"""
This scrip constructs the data base composed of fake images
created by the generator, and saves it to ~/SRTM_data/Fake.
--------
It creates as much images as in real images
--------
Requirements
--------
Before running this script make sure :
    - You have updated GENERATOR_ID
"""
import threading
import os
from pathlib import Path
import torch
import numpy as np
from networks.generator.generator import Generator, NOISE_LAYERS


class Generation(threading.Thread):
    """
    Generate charge_coeur images and register them
    """
    def __init__(self, charge_core, generator, path_fake_data, epoch):
        """
        Create the instance Generation
        """
        super().__init__()
        self.charge_coeur = charge_core
        self.generator = generator
        self.path_fake_data = path_fake_data
        self.epoch = epoch

    def run(self):
        """
        Execute the purpose of the class
        """
        # Generate initial noise
        noise = torch.randn(self.charge_coeur, NOISE_LAYERS, 1, 1)
        conditions = torch.randn(self.charge_coeur, 2)
        # Generate fake image with G
        fake = self.generator(noise, conditions).detach()
        # Register the images generated so that the discriminator can train itself on those images.
        register(fake, self.path_fake_data, self.epoch)


def register(images, path, num_epoch):
    """
    Register images in the folder where path leads
    """
    nb_images = len(images)

    # We register all the images
    for num_image in range(nb_images):
        path_image = path / str(num_image + num_epoch * nb_images)
        tensor_image = images[num_image][0]
        numpy_image = np.array(tensor_image)
        np.save(path_image, numpy_image)


def erase_folder(path):
    """
    Erase the content of the folder where path leads
    """
    name_images = os.listdir(path)
    for image in name_images:
        path_image = path / image
        os.remove(path_image)


if __name__ == "__main__":
    # Parameters
    NB_IMAGE_GENERATED = 20
    NB_CORE = 4
    NB_CHARGE_CORE = NB_IMAGE_GENERATED // NB_CORE

    PATH_FAKE = Path("../SRTM_data/Fake")
    PATH_FAKE.mkdir(parents=True, exist_ok=True)
    GENERATOR_ID = 4

    # Erase the data in path
    erase_folder(PATH_FAKE)

    # --- Creation of the networks --- #
    NET_G = Generator(0)

    # --- Loading the optimized generator --- #
    PATH_TRAIN_G = Path("../networks_data/generator/")
    END_PATH = str(GENERATOR_ID) + ".pth"
    NET_G.load_state_dict(torch.load(PATH_TRAIN_G / END_PATH))

    # --- Generate the images --- #
    # Multithreading
    REGISTRATION_1 = Generation(NB_CHARGE_CORE, NET_G, PATH_FAKE, 0)
    REGISTRATION_2 = Generation(NB_CHARGE_CORE, NET_G, PATH_FAKE, 1)
    REGISTRATION_3 = Generation(NB_CHARGE_CORE, NET_G, PATH_FAKE, 2)
    REGISTRATION_4 = Generation(NB_CHARGE_CORE, NET_G, PATH_FAKE, 3)

    REGISTRATION_1.start()
    REGISTRATION_2.start()
    REGISTRATION_3.start()
    REGISTRATION_4.start()
