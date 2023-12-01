from abc import ABC, abstractmethod

import lightning.pytorch as pl


class SpherinatorModule(ABC, pl.LightningModule):
    """
    Abstract base class for all spherinator modules to ensure that all methods for hipster are implemented.
    """

    @abstractmethod
    def get_input_size(self):
        """Returns the size of the images the model takes as input and generates as output."""

    @abstractmethod
    def project(self, images):
        """Returns the coordinates of the images in the latent space.

        Args:
            images (Tensor): Input images.
        """

    @abstractmethod
    def reconstruct(self, coordinates):
        """Reconstructs the images from the coordinates in the latent space.

        Args:
            coordinates (Tensor): Coordinates in the latent space.
        """

    @abstractmethod
    def reconstruction_loss(self, images, reconstructions):
        """Calculate the difference between the images and the reconstructions.

        Args:
            images (Tensor): Input images.
            reconstructions (Tensor): Reconstructed images.
        """
