"""This file executes unittests.
We do not pretend that it tests all our code."""

import unittest
from pathlib import Path
from random import randint
import os
import numpy as np
import torch
from src.networks.discriminator.shuffle_data_set_discriminator import shuffle_data_set_d
from src.networks.discriminator.discriminator import Discriminator
from src.pre_processing.renormalizing import normalize, table_to_list, interesting_image
from src.pre_processing.renormalizing import PERCENTAGE


class DiscriminatorTest(unittest.TestCase):
    """Integration tests for the discriminator."""
    def setUp(self):
        """Constructs the discriminator and the dataset."""
        # -- Construction of the discriminator -- #
        self.net = Discriminator(100)

        # -- Construction of the data_set -- #
        path_to_image = Path("SRTM_data/Test/Real")
        data_set = shuffle_data_set_d(path_to_image, path_to_image)[0]
        data_set = np.array(data_set)
        data_set = torch.from_numpy(data_set)
        size_batch = 1
        dim = data_set.shape
        nb_images = (1 // size_batch) * size_batch
        data_set = data_set[:nb_images].view(-1, size_batch, 1, dim[-2], dim[-1])
        self.data_set = data_set.to(torch.float32)

    def test_forward(self):
        """Tests if forward returns an object in the right format."""
        result = self.net(self.data_set[0])
        proba = self.net.soft_max(result.detach())

        # -- We verify the size of the result -- #
        self.assertEqual(result.shape[1], 2)

        # -- We verify the elements of the result -- #
        self.assertGreaterEqual(int(proba[0][0]), 0)
        self.assertLessEqual(int(proba[0][0]), 1)
        self.assertGreaterEqual(int(proba[0][1]), 0)
        self.assertLessEqual(int(proba[0][1]), 1)


class SuffledataTest(unittest.TestCase):
    """Integration tests for shuffle_data_set_discriminator."""
    def setUp(self):
        """Creates the paths and counts the nomber of real
        and fake images."""
        self.path_test_real = Path("SRTM_data/Real")
        self.path_test_fake = Path("SRTM_data/Fake")
        self.nb_fake = len(os.listdir(self.path_test_real))
        self.nb_real = len(os.listdir(self.path_test_fake))
        self.nb_images = self.nb_real + self.nb_fake

    def test_suffle(self):
        """Verifies if shuffle return the right number of images
        and if the labels are in {0, 1}."""
        (data_set, labels) = shuffle_data_set_d(self.path_test_real, self.path_test_fake)
        self.assertEqual(len(data_set), self.nb_images)
        for label in labels:
            self.assertIn(label, [0, 1])


class RenormalizingTest(unittest.TestCase):
    """Integration tests and unittests for renormalizing."""
    def setUp(self):
        """ Construct initial random array matrix."""
        self.nb_lines = randint(10, 100)
        self.nb_colons = randint(10, 100)
        random_image = np.zeros((self.nb_lines, self.nb_colons))
        for line in range(self.nb_lines):
            random_image[line, :] = np.array([randint(0, 255) for i in range(self.nb_colons)])
        self.random_matrix = random_image

    def test_normalize(self):
        """ Verifies if normalize is correct."""
        normalize_image = normalize(self.random_matrix)

        for line in range(self.nb_lines):
            for colon in range(self.nb_colons):
                self.assertGreaterEqual(normalize_image[line, colon], -1)
                self.assertLessEqual(normalize_image[line, colon], 1)

    def test_table_to_list(self):
        """Verifies that an array table is well converted in initial list."""
        list_matrix = table_to_list(self.random_matrix)
        self.assertEqual(len(list_matrix), self.nb_lines * self.nb_colons)
        index_line = randint(0, self.nb_lines - 1)
        index_colon = randint(0, self.nb_colons - 1)
        list_element = list_matrix[index_colon + index_line * self.nb_colons]
        matrix_element = self.random_matrix[index_line, index_colon]
        self.assertEqual(list_element, matrix_element)

    def test_interessting_image(self):
        """Verifies that an image that has more
        that 'pourcentage' pixel of the same color is rejected."""
        image_size = randint(10, 10)
        # We create an image that is on the limit of being interesting
        interesting_matrix = np.zeros((image_size, image_size))
        pourcentage_line = int(image_size * PERCENTAGE)
        for line in range(image_size):
            if line < pourcentage_line:
                new_line = np.array([0] * image_size)
            else:
                new_line = np.array([line * image_size + i for i in range(image_size)])
            interesting_matrix[line, :] = new_line

        # We create an image that in not interesting
        not_interesting_matrix = interesting_matrix.copy()
        not_interesting_matrix[-1, -1] = 0

        # We apply interesting_image to those images
        self.assertTrue(interesting_image(interesting_matrix))
        self.assertFalse(interesting_image(not_interesting_matrix))


if __name__ == '__main__':
    # -- Unittests -- #
    unittest.main()
