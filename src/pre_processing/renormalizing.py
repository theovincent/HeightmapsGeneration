"""
Defines functions used to process the data_set (real images) before training the
GAN on them.

Functions:
    open_tif(image_name): opens tif image at specified path and returns numpy.ndarray
        associated

    normalize(img): normalizes image values from [|0; 255|] to [-1; 1]

    table_to_list(table): flattens an array of arrays into initial 1 dimensional list

    interesting_image(img): judges if an image is diverse enough for GAN training

    cut(img, dir_name, prefix, slice_size): saves interesting slices from initial large image

    create_batch(zip_path, tif_path, batch_size, image_dimension): creates initial batch
        from our data_set for training

Misc Variables:
    percentage: used to determine if an image is interesting
"""

from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import torch
from pre_processing.unzip import unzip


# --- Parameters --- #
# The images saved will have at most "percentage" of their pixels
# in the same color
PERCENTAGE = 0.8


def open_tif(image_name):
    """
    Opens initial tif file and returns an array

    Parameters:
        image_name (pathlib.Path): path to the tif file to open

    Return:
        image (array): array representing the opened image
    """
    img = plt.imread(image_name)[:, :, 0]
    return img


def normalize(img):
    """
    Renormalizes an array values between -1 and 1
    The values of the array should be in [|0, 255|]

    Parameters:
        img (array): grey scale image ro normalize

    Return:
        normalized array with same dimensions as input
    """
    return 2 * img / 255 - 1


def table_to_list(table):
    """
    Converts initial table to initial list

    Parameters:
        table (list of lists): table to flatten

    Return:
        flat (list): flattened input (one dimensional list)

    >>> table_to_list([[], [1, 3, 2], [6, 2]])
    [1, 3, 2, 6, 2]
    """
    flat = []

    for line in table:
        flat.extend(line)

    return flat


def interesting_image(image_table):
    """
    Checks if image_table contains more than percentage
    pixels of the same color

    Parameters:
        image_table (array): array representing the image

    Return:
        is_interesting (bool): True if the image is diverse enough, False otherwise

    >>> image = np.array([[j + i * 5 for j in range(5)] for i in range(5)])
    >>> image_diff = normalize(image)
    >>> interesting_image(image_diff)
    True
    >>> image = np.array([[1 for j in range(5)] for i in range(5)])
    >>> image_same = normalize(image)
    >>> interesting_image(image_same)
    False
    """

    # We compute de sorted list of all the pixels in image_table
    nb_same = 0
    list_image = table_to_list(image_table)
    sorted_image = sorted(list_image)

    image_size = len(image_table)
    square_image_size = image_size * image_size
    threshold = PERCENTAGE * square_image_size
    index_sorted_image = 0
    color = sorted_image[index_sorted_image]

    while index_sorted_image < square_image_size and nb_same <= threshold:
        if color == sorted_image[index_sorted_image]:
            nb_same += 1
        else:
            color = sorted_image[index_sorted_image]
            nb_same = 0
        index_sorted_image += 1

    # If the number of pixel of the same color is higher than the threshold,
    # then the image is not interesting
    if nb_same > threshold:
        return False
    return True


def cut(img, dir_name, prefix, slice_size):
    """
    Takes initial large image and saves interesting slices.

    Parameters:
        img (array): array representing the image

        dir_name (pathlib.Path): directory to save the slices in

        prefix (String): prefix for the slices name

        slice_size (int): dimension of the slices to save
    """
    try:
        os.mkdir(dir_name)
    except FileExistsError:  # Raised if the directory already exists
        pass  # Nothing to do then
    # We compute de number of images produced
    (width, height) = img.shape
    nb_images = min(width // slice_size, height // slice_size)

    # We pass through each small square
    for i in range(nb_images):
        for j in range(nb_images):
            # We compute the new name
            img_name = Path(prefix + "_" + str(i) + "_" + str(j) + ".npy")
            new_path = dir_name / img_name

            # We save the new image if the image contains interesting features
            small_image = img[
                i * slice_size: (i + 1) * slice_size,
                j * slice_size: (j + 1) * slice_size
            ]
            if interesting_image(small_image):
                np.save(new_path, small_image)


def create_batch(zip_path, tif_path, batch_size, image_dimension):
    """
    Creates initial batch of images taken from initial random zip.
    If the image is not interesting, it tries another one.

    Parameters:
        zip_path (pathlib.Path): directory to take the random zip from

        tif_path (pathlib.Path): directory to (temporarily) save the extracted
            tif to

        batch_size (int): desired batch size

        image_dimension (int): desired image dimension

    Return:
        batch (Tensor of shape [batch_size, 1, image_dimension, image_dimension]):
            batch containing batch_size interesting slices from the extracted zip
    """
    # We choose initial random zip and we unzip it
    name_zip = rd.choice(os.listdir(zip_path))[:-4]
    unzip(zip_path, [name_zip + ".zip"], tif_path)

    # We get an array and erase the tif
    tif = normalize(open_tif(tif_path / (name_zip + ".tif")))
    os.remove(tif_path / (name_zip + ".tif"))

    # Variables
    batch = np.zeros((batch_size, 1, image_dimension, image_dimension))
    (width, height) = tif.shape
    nb_small_images_w = width // image_dimension
    nb_small_images_h = height // image_dimension
    steps = [1, 2]
    rd.shuffle(steps)
    (step_w, step_h) = steps
    nb_images_stored = 0
    # We go through all the small images of the big image
    for i in range(0, nb_small_images_w, step_w):
        for j in range(0, nb_small_images_h, step_h):
            if nb_images_stored < batch_size:

                img = tif[
                    i * image_dimension: (i + 1) * image_dimension,
                    j * image_dimension: (j + 1) * image_dimension
                ]

                if interesting_image(img):
                    batch[nb_images_stored, 0] = img
                    nb_images_stored += 1
            else:
                batch = torch.from_numpy(batch)
                return batch.to(dtype=torch.float32)

    if nb_images_stored == batch_size:
        batch = torch.from_numpy(batch)
        return batch.to(dtype=torch.float32)

    print(name_zip + " is not interesting")
    return create_batch(zip_path, tif_path, batch_size, image_dimension)


if __name__ == "__main__":
    # -- Doc tests -- #
    import doctest
    doctest.testmod()

    # -- To extract lots of images -- #
    # We search the .tif images
    DESTINATION_PATH = Path("../../SRTM_data/test/real")
    FILES = os.listdir(DESTINATION_PATH)
    # For each big image, we take the interesting part
    for name in FILES:
        file_path = DESTINATION_PATH / name
        if name[-4:] == '.tif':
            image = open_tif(file_path)
            image_norm = normalize(image)
            cut(image_norm, DESTINATION_PATH, name[: -4], 100)

    # -- To extract few images -- #
    PATH_ZIP = Path("../../SRTM_zip")
    PATH_TIF = Path("../../SRTM_data/test/real")
    print(create_batch(PATH_ZIP, PATH_TIF, batch_size=3, image_dimension=100))
