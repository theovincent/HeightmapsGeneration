"""
Module defining the discriminator class of initial GAN for heightmaps generation.

Running it as initial script saves initial random (not trained) discriminator state at
networks_data/discriminator/0.pth.

Classes:
    Discriminator
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy import floor


class Discriminator(nn.Module):
    """
    The discriminator is a Convolutional Neural Network that decides
    whether each instance of data is a real image or not.

    Attributes:
        dim (int): dimension of input images (Tensor of shape [dim x dim]).

        final_dim (int): dimension of input exiting the convolution layers.

        conv1, conv2, conv3 (torch.nn.modules.conv.Conv2d): convolutions for each layer.

        fc1, fc2, fc3 (torch.nn.modules.linear.Linear): affine functions
            applied after the convolutions layers.

        pool (torch.nn.modules.pooling.MaxPool2d): 2 dimensional max pooling of size 2x2.

        soft_max (torch.nn.modules.activation.Softmax): softmax function,
            computes probability for each class from network output.

    Methods:
        forward(x): computes output from input image x.

        test(data_set, label): displays discriminator performances
            on data_set according to labels.

        custom_training(data_set, labels): trains network on data_set according to labels.
    """

    def __init__(self, dim):
        """
        Creates the discriminator : the layers of operations to apply.

        Parameters:
            dim (int): dimension of input images (Tensor of shape [dim x dim]).
        """
        super(Discriminator, self).__init__()

        self.dim = dim
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        self.final_dim = int(floor(dim / 8 - 7 / 2))

        # Linear layers
        self.fc1 = nn.Linear(self.final_dim * self.final_dim * 32, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 2)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forwards input image through the networks layers

        Parameters:
            x (Tensor of shape [batch_size, 1, self.dim, self.dim]): input image

        Return:
            energies (Tensor of shape [batch_size, 2]): energies associated
                with each class (Real or Fake)
        """
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.pool(f.relu(self.conv3(x)))

        x = x.view(-1, self.final_dim * self.final_dim * 32)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def test(self, data_set, labels):
        """
        Prints the training of the discriminator

        Parameters:
            data_set (Tensor of shape [nb_batchs, batch_size, 1, self.dim, self.dim]):
                images to be tested

            labels (Tensor of shape [nb_batchs, batch_size, 1]): labels corresponding
                to each image in data_set (0 for real, 1 for fake)
        """

        with torch.no_grad():
            nb_image = data_set.shape[0]

            # stats
            # format : actual_predicted
            real_real = 0
            fake_real = 0
            real_fake = 0
            fake_fake = 0

            avg_fake = 0
            avg_real = 0

            nb_real = 0
            nb_fake = 0

            for i in range(nb_image):
                image = data_set[i]
                label = labels[i]

                probas = self.soft_max(self(image))

                if label == 0:
                    nb_real += 1
                    avg_real += probas[0]

                    if probas[0] > 0.5:
                        real_real += 1
                    else:
                        real_fake += 1
                else:
                    nb_fake += 1
                    avg_fake += probas[1]

                    if probas[1] > 0.5:
                        fake_fake += 1
                    else:
                        fake_real += 1

                print("Real images predictions:", real_real / nb_real,
                      "real,", real_fake / nb_real,
                      "fake, average proba of real:", avg_real / nb_real
                      )
                print("fake images predictions:", fake_real / nb_fake,
                      "real,", fake_fake / nb_fake,
                      "fake, average proba of fake:", avg_fake / nb_fake,
                      )

    def custom_training(self, data_set, labels):
        """
        Trains the discriminator.

        Parameters:
            data_set (Tensor of shape [nb_batchs, batch_size, 1, self.dim, self.dim]):
                images to be trained on.

            labels (Tensor of shape [nb_batchs, batch_size, 1]): labels corresponding
                to each image in data_set (0 for real, 1 for fake).
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        nb_batchs = data_set.shape[0]

        for i in range(nb_batchs):
            batch = data_set[i]
            label = labels[i]

            optimizer.zero_grad()

            outputs = self(batch)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()


if __name__ == "__main__":
    NET = Discriminator(100)
    torch.save(
        NET.state_dict(),
        Path("../../../networks_data/discriminator/0.pth")
    )
