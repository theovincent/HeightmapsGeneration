from PIL import Image
from src.pre_processing.renormalizing import interesting_image
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import torch
from pre_processing.unzip import unzip
import imageio


def show(name_file):
    path = Path("./SRTM_data/Test")
    file_path = path / name_file
    im = Image.open(file_path)
    im.show()
    return im


def open_tif(image_name):
    """Opens a tif file and returns an array"""
    img = plt.imread(image_name)[:, :, 0]
    return img


def normalize(img):
    """Renormalize an array between -1 and 1
    The values of the array should be in [0, 255]"""
    return 2 * img / 255 - 1


def create_batch(zip_path, tif_path, path_save, image_dimension):
    """
    Creates a batch of images taken from a zip.
    If the image is not interessting, it takes another one.
    """
    # We choose a random zip and we unzip it
    name_zip = rd.choice(os.listdir(zip_path))[:-4]
    unzip(zip_path, [name_zip + ".zip"], tif_path)

    # We get an array and erase the tif
    tif = normalize(open_tif(tif_path / (name_zip + ".tif")))
    os.remove(tif_path / (name_zip + ".tif"))

    # Variables
    (w, h) = (len(tif), len(tif[0]))
    nb_small_images_w = w // image_dimension
    nb_small_images_h = h // image_dimension
    # We go through all the small images of the big image
    for i in range(0, nb_small_images_w):
        for j in range(0, nb_small_images_h):
            img = tif[
                  i * image_dimension: (i + 1) * image_dimension,
                  j * image_dimension: (j + 1) * image_dimension
            ]
            if interesting_image(img):
                print(img.shape)
                path_save_c = path_save / "{}_{}.png".format(i, j)
                imageio.imwrite(path_save_c, img)


if __name__ == "__main__":

    zip_path = Path("./SRTM_data/Test/Zip")
    save_path = Path("./SRTM_data/Test")
    create_batch(zip_path, zip_path, save_path, 100)
