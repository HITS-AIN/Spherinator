from typing import Optional

import lightning.pytorch as pl
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as functional
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam

from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder


class VariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        h_dim: int = 256,
        z_dim: int = 3,
        beta: float = 1.0,
    ):
        """VariationalAutoencoder initializer

        Args:
            encoder (Optional[nn.Module], optional): encoder model. Defaults to None.
            decoder (Optional[nn.Module], optional): decoder model. Defaults to None.
            h_dim (int, optional): dimension of the hidden layers. Defaults to 256.
            z_dim (int, optional): dimension of the latent representation. Defaults to 3.
            beta (float, optional): factor for beta-VAE. Defaults to 1.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        if encoder is None:
            encoder = ConvolutionalEncoder(latent_dim=h_dim)
        if decoder is None:
            decoder = ConvolutionalDecoder(latent_dim=h_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.beta = beta

        self.example_input_array = self.encoder.example_input_array

        self.fc_location = nn.Linear(h_dim, z_dim)
        self.fc_scale = nn.Linear(h_dim, 1)
        self.fc2 = nn.Linear(z_dim, h_dim)

        with torch.no_grad():
            self.fc_scale.bias.fill_(1.0e3)

    def encode(self, x):
        x = self.encoder(x)
        z_location = self.fc_location(x)
        z_location = torch.nn.functional.normalize(z_location, p=2.0, dim=1)
        # SVAE code: the `+ 1` prevent collapsing behaviors
        z_scale = F.softplus(self.fc_scale(x)) + 1

        return z_location, z_scale

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = self.decoder(x)
        return x

    def reparameterize(self, z_location, z_scale):
        q_z = PowerSpherical(z_location, z_scale)
        p_z = HypersphericalUniform(self.z_dim, device=z_location.device)
        return q_z, p_z

    def forward(self, x):
        z_location, z_scale = self.encode(x)
        q_z, p_z = self.reparameterize(z_location, z_scale.squeeze())
        z = q_z.rsample()
        recon = self.decode(z)
        return (z_location, z_scale), (q_z, p_z), z, recon

    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            crop = functional.center_crop(batch, [self.crop_size, self.crop_size])
            scaled = functional.resize(
                crop, [self.input_size, self.input_size], antialias=True
            )

        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(scaled)
        loss_recon = self.reconstruction_loss(scaled, recon)

        for i in range(1, self.rotations):
            with torch.no_grad():
                rotate = functional.rotate(
                    batch, 360.0 / self.rotations * i, expand=False
                )
                crop = functional.center_crop(rotate, [self.crop_size, self.crop_size])
                scaled = functional.resize(
                    crop, [self.input_size, self.input_size], antialias=True
                )

            loss_recon = torch.min(loss_recon, self.reconstruction_loss(scaled, recon))

        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z) * self.beta
        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_recon", loss_recon, prog_bar=True)
        self.log("loss_KL", loss_KL)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        self.log("mean(z_location)", torch.mean(z_location))
        self.log("mean(z_scale)", torch.mean(z_scale))
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)
