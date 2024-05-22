from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch
import torchvision.transforms.v2.functional as functional


class SpherinatorModule(ABC, pl.LightningModule):
    """
    Abstract base class for all spherinator modules to ensure that all methods for hipster are implemented.
    """

    @abstractmethod
    def get_input_size(self):
        """Returns the size of the images the model takes as input and generates as output."""

    def find_best_rotation(self, batch):
        """Find the rotated image with the lowest reconstruction loss.

        Args:
            batch (Tensor): Input batch of images.

        Returns:
            best_scaled_image (Tensor): The best scaled image.
            best_rotations (Tensor): The rotation for each image.
            best_coordinates (Tensor): The coordinates for each image.
            best_recon (Tensor): The reconstruction loss for each image.
        """
        with torch.no_grad():
            best_recon = torch.ones(batch.shape[0], device=batch.device) * 1e10
            best_recon_idx = torch.zeros(batch.shape[0], device=batch.device)
            best_rotations = torch.zeros(batch.shape[0], device=batch.device)
            best_scaled_image = torch.zeros(
                (batch.shape[0], batch.shape[1], self.input_size, self.input_size),
                device=batch.device,
            )
            best_coordinates = torch.zeros(
                (batch.shape[0], self.z_dim), device=batch.device
            )

            for i in range(self.rotations):
                rotate = functional.rotate(
                    batch, 360.0 / self.rotations * i, expand=False
                )
                crop = functional.center_crop(rotate, [self.crop_size, self.crop_size])
                scaled = functional.resize(
                    crop, [self.input_size, self.input_size], antialias=True
                )

                coordinates = self.project(scaled)
                reconstruction = self.reconstruct(coordinates)
                loss_recon = self.reconstruction_loss(scaled, reconstruction)

                best_recon_idx = torch.where(loss_recon < best_recon)
                best_recon[best_recon_idx] = loss_recon[best_recon_idx]
                best_scaled_image[best_recon_idx] = scaled[best_recon_idx]
                best_coordinates[best_recon_idx] = coordinates[best_recon_idx]

                for j in range(batch.shape[0]):
                    if j in best_recon_idx[0]:
                        best_rotations[j] = i

            best_rotations *= 360.0 / self.rotations
            return best_scaled_image, best_rotations, best_coordinates, best_recon

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
