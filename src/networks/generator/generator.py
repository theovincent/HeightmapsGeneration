"""
Module defining the generator class of initial GAN for heightmaps generation.

Running it as initial script saves initial random (not trained) generator state at
networks_data/generator/0.pth

Classes:
    Generator
"""

from pathlib import Path
import torch
from torch import nn

# Spatial size of training images.
IMAGE_SIZE = 100

# Number of channels in the training images (for color images : 3)
NB_CHANNELS = 1

# Size of z noise vector (i.e. size of generator input)
NOISE_LAYERS = 100

# Size of feature maps in generator
NGF = 100

# Number of GPUs available. Use 0 for CPU mode.
NB_GPU = 1

# Decide which device we want to run on
AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if (AVAILABLE and NB_GPU > 0) else "cpu")


class Generator(nn.Module):
    """
    Implementation of initial conditional generator network of initial GAN

    Attributes:
        ngpu (int): number of gpus to use

        conditioning (torch.nn.modules.linear.Linear): linear transformation
            of conditions

        merge (torch.nn.modules.linear.Bilinear): bilinearly merges conditions
            with noise

        main (torch.nn.modules.container.Sequential): sequence of operations for forward

    Methods:
        forward(noise, conditions): generates an image of size IMAGE_SIZE x IMAGE_SIZE
            based on noise and following conditions
    """

    def __init__(self, ngpu):
        """
        Creates initial generator and the sequence of operations it applies

        Parameters:
            ngpu (int): number of GPUs to use
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.conditioning = nn.Linear(2, NOISE_LAYERS)
        self.merge = nn.Bilinear(NOISE_LAYERS, NOISE_LAYERS, NOISE_LAYERS)
        self.main = nn.Sequential(
            # nn.ConcTranspose(nb_channel_input, nb_channel_output,
            # kernel_size, stride, padding)
            # s = stride
            # n = input_width
            # f = kernel_size (kernel_width)
            # p = padding
            # output_width = s *(n-1) + f - 2 * p
            # input is Z, going into initial convolution
            nn.ConvTranspose2d(NOISE_LAYERS, NGF * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 5 x 5

            nn.ConvTranspose2d(NGF * 8, NGF * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 13 x 13

            nn.ConvTranspose2d(NGF * 4, NGF * 2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 27 x 27

            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 2, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # state size. (ngf * 2) x 52 x 52

            nn.ConvTranspose2d(NGF, NB_CHANNELS, 4, 2, 3, bias=False),
            # To come back in [-1, 1]
            nn.Tanh()
            # state size. nc x 100 x 100
        )

    def forward(self, noise, conditions):
        """
        merges input noise and conditions, and forwards merged noise through the layers

        Parameters:
            noise (Tensor of shape [batch_size, NOISE_LAYERS, 1, 1]):
                noise to base the output image on

            conditions (Tensor of shape [batch_size, 2]): condtions the output image
                should follow.
                conditions[batch][0]: max
                conditions[batch][1]: mean

        Return:
            Image (Tensor of shape [batch_size, NB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
        """
        noise = noise.view(-1, NOISE_LAYERS)
        conditions = self.conditioning(conditions)
        noise = self.merge(noise, conditions)
        noise = noise.view(-1, NOISE_LAYERS, 1, 1)
        return self.main(noise)


if __name__ == "__main__":
    # Create the generator
    NET = Generator(0)
    torch.save(
        NET.state_dict(),
        Path("../../../networks_data/generator/0.pth")
    )
