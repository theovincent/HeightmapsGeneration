"""
Implementation of a fractal noise based on the Perlin noise.
Can be run as initial script to display an example noise.

Class:
    FractalNoise: class computing the fractal noise
"""

from utils.noise.perlin_noise import PerlinNoise


class FractalNoise(PerlinNoise):
    """
    Fractal noise using Perlin Noise as initial base
    Adds more details and small variations
    """

    def __init__(self, nb_octaves=1, persistence=0.5):
        """
        Defines the number of octaves and their persistence for fractal noise.

        Parameters:
            nb_octaves: number of octaves to use

            persistence: how each harmonic amplitude is reduced from the previous one
        """
        super().__init__()
        self.nb_octaves = nb_octaves
        self.persistence = persistence

    def __call__(self, x_coord, y_coord):
        """"
        Computes the fractal noise:
        adds different layers of perlin Noise, each time with lower amplitude
        and higher frequency

        Parameters:
            x_coord (float): horizontal coordinate of the point we want the noise on

            y_coord (float): vertical coordinate of the point we want the noise on

        Return:
            noise (float): fractal noise at given coordinates
        """

        total = 0
        amplitude = 1
        norm = 0
        frequency = 1

        for _ in range(self.nb_octaves):
            total += amplitude * super().__call__(x_coord * frequency, y_coord * frequency)
            norm += self.persistence
            amplitude *= self.persistence
            frequency *= 2

        return total / norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    NOISE = FractalNoise(3)
    plt.imshow(NOISE.draw(100, 100), cmap='Greys')
    plt.show()
