"""
Contains useful loss functions of conditional GAN

Functions:
    loss(img, conditions): computes the error between the image and the conditions
        it is supposed to respect
"""

import torch

def loss(img, conditions):
    """
    Computes the distance between imposed conditions and the actual properties of
    the image.

    Parameters:
        img (Tensor of shape [batch_size, 1, dim, dim]): batch of images
            that should follow certain condtions

        conditions (Tensor fof shape [batch_size, 2]): conditions imposed on
            the images.
            conditions[batch][0]: max
            conditions[batch][1]: mean

    Return:
        dist (float): norm of the distances between conditions and actual properties
            for each image in the batch
    """
    loss_max = torch.abs(
        img.max(-1)[0].max(-1)[0].view(-1) - conditions[:, 0]
    ).norm()
    loss_avg = torch.abs(
        img.mean(-1)[0].mean(-1)[0].view(-1) - conditions[:, 1]
    ).norm()
    return loss_max + loss_avg
