import os
import sys
import math

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional
from torch.optim import Adam

from .spherinator_module import SpherinatorModule

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../external/s-vae-pytorch/"))
from hyperspherical_vae.distributions import HypersphericalUniform, VonMisesFisher


class RotationalVariationalAutoencoder(SpherinatorModule):
    def __init__(
        self,
        h_dim: int = 256,
        z_dim: int = 2,
        distribution: str = "normal",
        image_size: int = 91,
        rotations: int = 36,
        beta: float = 1.0,
        spherical_loss_weight: float = 1e-4,
    ):
        """
        RotationalVariationalAutoencoder initializer

        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        :param image_size: size of the input images
        :param rotations: number of rotations
        :param beta: factor for beta-VAE
        :param spherical_loss_weight: weight of the spherical loss
        """
        super().__init__()
        self.save_hyperparameters()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.distribution = distribution
        self.image_size = image_size
        self.rotations = rotations
        self.beta = beta
        self.spherical_loss_weight = spherical_loss_weight

        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.input_size = 64
        self.total_input_size = self.input_size * self.input_size * 3

        self.example_input_array = torch.randn(1, 3, self.input_size, self.input_size)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2, padding=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(5, 5), stride=2, padding=2
        )

        self.fc1 = nn.Linear(256 * 4 * 4, h_dim)

        self.fc_mean = nn.Linear(h_dim, z_dim)
        if self.distribution == "normal":
            # compute mean and std of the normal distribution
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif self.distribution == "vmf":
            # compute mean and concentration of the von Mises-Fisher
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplementedError

        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 256 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=(4, 4), stride=2, padding=1
        )
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=16, out_channels=3, kernel_size=(5, 5), stride=1, padding=2
        )

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 256 * 4 * 4)
        x = F.tanh(self.fc1(x))

        z_mean = self.fc_mean(x)
        if self.distribution == "normal":
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == "vmf":
            z_mean = torch.nn.functional.normalize(z_mean, p=2.0, dim=1)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1.0e-6
        else:
            raise NotImplementedError

        return z_mean, z_var

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x

    def reparameterize(self, z_mean, z_var):
        if self.distribution == "normal":
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(
                torch.zeros_like(z_mean), torch.ones_like(z_var)
            )
        elif self.distribution == "vmf":
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, device=z_mean.device)
        else:
            raise NotImplementedError
        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        recon = self.decode(z)
        return (z_mean, z_var), (q_z, p_z), z, recon

    def spherical_loss(self, coordinates):
        return torch.square(1 - torch.sum(torch.square(coordinates), dim=1))

    def training_step(self, batch, batch_idx):
        best_recon = torch.ones(batch.shape[0], device=batch.device) * 1e10
        best_scaled = torch.zeros(
            (batch.shape[0], batch.shape[1], self.input_size, self.input_size),
            device=batch.device,
        )

        with torch.no_grad():
            for i in range(self.rotations):
                rotate = functional.rotate(
                    batch, 360.0 / self.rotations * i, expand=False
                )
                crop = functional.center_crop(rotate, [self.crop_size, self.crop_size])
                scaled = functional.resize(
                    crop, [self.input_size, self.input_size], antialias=True
                )

                (_, _), (_, _), _, recon = self.forward(scaled)
                loss_recon = self.reconstruction_loss(scaled, recon)

                best_recon_idx = torch.where(loss_recon < best_recon)
                best_recon[best_recon_idx] = loss_recon[best_recon_idx]

                best_scaled[best_recon_idx] = scaled[best_recon_idx]

        (z_mean, z_var), (q_z, p_z), _, recon = self.forward(best_scaled)

        loss_recon = self.reconstruction_loss(best_scaled, recon)

        if self.distribution == "normal":
            loss_KL = (
                self.beta
                * torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            )
        elif self.distribution == "vmf":
            loss_KL = self.beta * torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            raise NotImplementedError

        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_recon", loss_recon, prog_bar=True)
        self.log("loss_KL", loss_KL)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        self.log("mean(z_mean)", torch.mean(z_mean))
        self.log("mean(z_var)", torch.mean(z_var))
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)

    def project(self, images):
        z_mean, _ = self.encode(images)
        return z_mean

    def reconstruct(self, coordinates):
        return self.decode(coordinates)

    def reconstruction_loss(self, images, reconstructions):
        return torch.sqrt(
            nn.MSELoss(reduction="none")(
                reconstructions.reshape(-1, self.total_input_size),
                images.reshape(-1, self.total_input_size),
            ).mean(dim=1)
        )
