"""
Defines initial class for the standard implementation of the 2D Perlin Noise.
Can be run as initial script to display an example noise.

Class:
    PerlinNoise: class for Perlin noise
"""

from numpy import floor, sqrt, zeros
import numpy.random as rand


class PerlinNoise:
    """
    Standard implementation of Perlin Noise in 2D

        Attributes:
            hash_tab (list): hash values to compute pseudo-random gradients

        Methods:
            grad(hash_value, x_coord, y_coord): computes the scalar product
                <grad(hash), (x_coord, y_coord)>

            fade(coord): computes the fade function for smoothness

            lerp(initial, final, lambda_): computes the linear interpolation
                between initial and final with weight lambda_

            self(x_coord, y_coord): computes perlin noise at give coordinates

            draw(height, width, x_off, y_off, x_incr, y_incr): draws an image with
                perlin noise, with initial coordinates offset by x_off and y_off
                and incrementing coordinates with x_incr and y_incr
    """

    def __init__(self):
        """
        Creates initial random hash_table to compute pseudo-random gradients
        """

        hash_tab = list(range(256))
        rand.shuffle(hash_tab)

        self.hash_tab = hash_tab + hash_tab

    @staticmethod
    def grad(hash_value, x_coord, y_coord):
        """
        Computes the scalar product of (x, y) and initial pseudo-random gradient
        given by the hash

        Parameters:
            hash_value (int): hash giving the gradient

            x_coord (float): relative horizontal coordinate of the point to
                compute the scalar product on

            y_coord (float): relative vertical coordinate of the point to
                compute the scalar product on

        Return:
            product (float): desired scalar product
        """

        grad_index = hash_value % 4

        if grad_index == 0:
            return x_coord + y_coord
        if grad_index == 1:
            return -x_coord + y_coord
        if grad_index == 2:
            return x_coord - y_coord

        return -x_coord - y_coord

    @staticmethod
    def fade(coord):
        """
        Fade function to preserve smoothness during interpolation

        Parameters:
            coord (float): relative coordinate we want to interpolate on

        Return:
            val (float): image of t by the fade polynomial function
        """

        return coord * coord * coord * (coord * (coord * 6 - 15) + 10)

    @staticmethod
    def lerp(initial, final, lambda_):
        """
        Computes the linear interpolation between a and b with coefficient
        lambda_

        Parameters:
            initial (float): initial coordinate for interpolation

            final (float): final coordinate for interpolation

            lambda_ (float): weight of the wanted point in [a, b]

        Return:
            val (float): linear interpolation of weight lambda_ in [a, b]
        """

        return initial + lambda_ * (final - initial)

    def __call__(self, x_coord, y_coord):
        """
        Computes the Perlin Noise algorithm at given (x, y) coordinates

        Parameters:
            x_coord (float): horizontal coordinate of the point we want the noise on

            y_coord (float): vertical coordinate of the point we want the noise on

        Return:
            noise (float): Perlin noise at given coordinates
        """

        # Absolute coordinates of the square (x, y) is in
        square_x = int(floor(x_coord)) % 256
        square_y = int(floor(y_coord)) % 256

        # Relative coordinates of (x, y) in that square
        relative_x = x_coord - floor(x_coord)
        relative_y = y_coord - floor(y_coord)

        # hash value of each corner of the square given by the hash table
        hash1 = self.hash_tab[self.hash_tab[square_x] + square_y]
        hash2 = self.hash_tab[self.hash_tab[square_x + 1] + square_y]
        hash3 = self.hash_tab[self.hash_tab[square_x] + square_y + 1]
        hash4 = self.hash_tab[self.hash_tab[square_x + 1] + square_y + 1]

        grad1 = self.grad(hash1, relative_x, relative_y)
        grad2 = self.grad(hash2, relative_x - 1, relative_y)
        grad3 = self.grad(hash3, relative_x, relative_y - 1)
        grad4 = self.grad(hash4, relative_x - 1, relative_y - 1)

        faded_x = self.fade(relative_x)
        faded_y = self.fade(relative_y)

        return self.lerp(  # interpolation along vertical axis (y)
            self.lerp(  # first interpolation along horizontal axis (x)
                grad1,
                grad2,
                faded_x
            ),
            self.lerp(  # second interpolation along horizontal axis (x)
                grad3,
                grad4,
                faded_x
            ),
            faded_y
        ) * sqrt(2)  # for renormalization, giving an output in [-1, 1]

    def draw(self, height, width, x_off=sqrt(2), y_off=sqrt(2), x_incr=0.11, y_incr=0.11):
        """
        Draws a 2D image with pixel values equal to Perlin noise

        Parameters:
            height (int): height of desired output

            width (int): width of desired output

            x_off (float): initial horizontal offset (to avoid 0 value of noise
                at integer coordinates

            y_off (float): initial vertical offset (to avoid 0 value of noise
                at integer coordinates

            x_incr (float): horizontal increment of noise between adjacent pixels

            y_incr (float): vertical increment of noise between adjacent pixels

        Return:
            img (array of shape [height, width]): array representing the image
                given by the perlin noise
        """
        img = zeros((height, width))

        y_coord = y_off
        for i in range(height):
            x_coord = x_off
            for j in range(width):
                img[i, j] = self(x_coord, y_coord)
                x_coord += x_incr
            y_coord += y_incr

        return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    NOISE = PerlinNoise()
    plt.imshow(NOISE.draw(100, 100), cmap='Greys')
    plt.show()
