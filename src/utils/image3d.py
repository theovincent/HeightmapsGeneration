"""
This program takes initial black and white png image
and displays an elevation map. This map has values
between 0 and max_value.

Functions:
    display_3d(image, title): transforms image in heigtmap and displays it

    mean_image(image): makes each pixel the mean of its neighbours

    decrease_grad(image): reduces spatial gradients too much higher than their neighbours

    neighbour(image, index_line, index_col): computes the sum of the neighbours of given pixel

    blur(image): blurs the image

    maximum(table): returns the max of a list of lists

    pixel_to_elevation(image): computes a heightmap from an image with values in [-1; 1]

    show_3d(image, title): displays given heightmap with given title
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# -- Parameters -- #
MAX_VALUE = 1
TITLE1 = 'generated elevation map converted into 3D and displayed with blurred correction'
TITLE2 = 'generated elevation map converted into 3D and displayed without correction'


def display_3d(image, title=TITLE2):
    """
    Shows the image in 3 dimensions.

    Parameters:
        image (array): heightmap to display

        title (String): title of the figure
    """

    axes = Axes3D(plt.figure())
    # We do not take the edges
    x_coord = np.array(range(1, len(image[0]) - 1))
    y_coord = np.array(range(1, len(image) - 1))

    def pixel(index_line, index_col):
        return image[index_line, index_col]

    (x_coord, y_coord) = np.meshgrid(x_coord, y_coord)
    z_coord = pixel(x_coord, y_coord)

    # We interpolate to smooth the elevation map
    # (x, y) = np.mgrid[0:1:100j, 0:1:100j]
    # tck = interpolate.bisplrep(x, y, z, s=0)
    # z = interpolate.bisplev(x[:, 0], y[0, :], tck)
    axes.plot_surface(
        x_coord, y_coord, z_coord,
        rstride=1, cstride=1, cmap='winter', edgecolor='none'
    )
    plt.title(title)


def mean_image(image):
    """
    For each pixel, makes the mean of the pixel that are near.
    For edge pixels, black pixel are added after the borders

    Parameters:
        image (array): image to apply the transformation to
    """
    image_ref = image.copy()
    (weight, height) = image.shape
    for i in range(1, weight - 1):
        for j in range(1, height - 1):
            new_pixel_w = image_ref[i - 1, j] + image_ref[i + 1, j]
            new_pixel_h = image_ref[i, j - 1] + image_ref[i, j + 1]
            image[i, j] = (new_pixel_h + new_pixel_w) / 4


def decrease_grad(image):
    """
    Decreases the gradient of the image if it is too high

    Parameters:
        image (array): image to apply the transformation to
    """
    image_ref = image.copy()
    (weight, height) = image.shape
    for i in range(1, weight - 1):
        for j in range(1, height - 1):
            grad_pixel_w = (image_ref[i + 1, j] - image_ref[i - 1, j]) ** 2
            grad_pixel_h = (image_ref[i, j + 1] - image_ref[i, j - 1]) ** 2
            if grad_pixel_w > grad_pixel_h:
                print("w")
                new_pixel_w = (image_ref[i - 1, j] + image_ref[i + 1, j]) / 2
                image[i, j] = new_pixel_w
            else:
                print("h")
                new_pixel_h = (image_ref[i, j - 1] + image_ref[i, j + 1]) / 2
                image[i, j] = new_pixel_h / 2


def neighbour(image, index_line, index_col):
    """
    Returns the sum of the pixel near image[index_line, index_col]

    Parameters:
        image (array): initial image

        index_line (int): index of the line of the pixel

        index_col (int): index of the column of the pixel

    Return:
        sum_pixel (float): sum of the neighbours

        nb_pixel_touched (int): number of neighbours
    """
    sum_pixel = 0
    (weight, height) = image.shape
    nb_pixel_touched = 0

    for index_h in range(-1, 1):
        for index_w in range(-1, 1):
            index_i = index_line + index_h
            index_j = index_col + index_w
            index_i_not_in_image = index_i < 0 or index_i >= height
            index_j_not_in_image = index_j < 0 or index_j >= weight
            if index_i_not_in_image or index_j_not_in_image:
                pass
            else:
                sum_pixel += image[index_i, index_j]
                nb_pixel_touched += 1

    return sum_pixel, nb_pixel_touched


def blur(image):
    """
    Put initial blur on the image

    Parameters:
        image (array): image to apply the transformation to
    """
    (weight, height) = image.shape
    image_ref = image.copy()
    for index_w in range(1, weight):
        for index_h in range(1, height):
            (sum_neighbor, nb_pixel) = neighbour(image_ref, index_h, index_w)
            image[index_w, index_h] = sum_neighbor / nb_pixel


def maximum(table):
    """
    Returns the maximum of the table.

    Parameters:
        table (list of lists):table we want the max of

    Return:
        max (int): max of input table

    >>> maximum([[0, 1], [9, 1]])
    9
    """
    max_table = table[0][0]

    for line in table:
        max_line = max(line)
        if max_line > max_table:
            max_table = max_line

    return max_table


def pixel_to_elevation(image):
    """
    Transfer the vector image to initial vector in [0, max_value]

    Parameters:
        image (array): image to apply the transformation to
    """
    max_image = maximum(image)
    min_image = -maximum(-image)

    return (max_image - image) / (max_image - min_image) * MAX_VALUE


def show_3d(image, title=TITLE1):
    """
    Displays the png image as initial 3 dimensions
    elevation map.

    Parameters:
        image (array): heightmap to display

        title (String): title of the figure
    """
    # Translates it to have the good format
    np_elevation = pixel_to_elevation(image)

    # We make the image smoother
    for _ in range(5):
        blur(np_elevation)

    # Prints it and saves it
    display_3d(np_elevation, title)


if __name__ == "__main__":
    # -- Doc test -- #
    import doctest
    doctest.testmod()

    # -- Find the path to the image -- #
    PATH_TO_FILE = Path("../../SRTM_data/Real")
    FILES = os.listdir(PATH_TO_FILE)
    PATH_TO_IMAGE = PATH_TO_FILE / FILES[0]
    # Gets the pixels
    IMAGE = np.load(PATH_TO_IMAGE)
    # Transforms into an array
    IMAGE = np.array(IMAGE)

    # -- Display the image and saves it -- #
    SAVE_PATH = Path('../../static/images/img_{}.png'.format(30))
    show_3d(IMAGE, SAVE_PATH)
    plt.show()
