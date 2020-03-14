"""
Modules defining initial tools to create initial training data set for the discriminator.

Functions:
    shuffle_data_set_d
"""

import random
import os
from pathlib import Path
import numpy as np


def shuffle_data_set_d(destination_path_real_images, destination_path_false_images):
    """
    Creates an array of real and fake images with the corresponding
    labels.

    Parameters:
        destination_path_real_images (pathlib.Path): directory to extract real images from

        destination_path_false_images (pathlib.Path): directory to extract fake images from

    Return:
        data_set (list): list of fake and real images, shuffled

        labels (list): list of labels for each image in data_set
            (0 for real, 1 for fake)

    >>> path_test_real = Path("../../../SRTM_data/test/Real")
    >>> path_test_fake = Path("../../../SRTM_data/test/Fake")
    >>> (data_set, label) = shuffle_data_set_d(path_test_real, path_test_fake)
    >>> len(data_set)
    2
    >>> len(label)
    2
    """
    real_images_names = os.listdir(destination_path_real_images)
    false_images_names = os.listdir(destination_path_false_images)

    real_images = []
    false_images = []
    label = []

    for name in real_images_names:
        real_images.append(np.load(destination_path_real_images / name))
        label.append(0)

    for name in false_images_names:
        false_images.append(np.load(destination_path_false_images / name))
        label.append(1)

    data_set = real_images + false_images
    zipped = list(zip(data_set, label))
    random.shuffle(zipped)
    data_set, label = zip(*zipped)

    return data_set, label


if __name__ == "__main__":
    # -- Doc tests -- #
    import doctest
    doctest.testmod()
