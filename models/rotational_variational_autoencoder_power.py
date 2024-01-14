import math

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam

from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder
from .spherinator_module import SpherinatorModule


class RotationalVariationalAutoencoderPower(SpherinatorModule):
    def __init__(
        self,
        encoder: object = ConvolutionalEncoder,
        decoder: object = ConvolutionalDecoder,
        h_dim: int = 256,
        z_dim: int = 2,
        image_size: int = 91,
        rotations: int = 36,
        beta: float = 1.0,
    ):
        """RotationalVariationalAutoencoderPower initializer

        Args:
            h_dim (int, optional): dimension of the hidden layers. Defaults to 256.
            z_dim (int, optional): dimension of the latent representation. Defaults to 2.
            image_size (int, optional): size of the input images. Defaults to 91.
            rotations (int, optional): number of rotations. Defaults to 36.
            beta (float, optional): factor for beta-VAE. Defaults to 1.0.
        """
        super().__init__()
        self.save_hyperparameters()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.image_size = image_size
        self.rotations = rotations
        self.beta = beta

        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.input_size = 128
        self.total_input_size = self.input_size * self.input_size * 3

        self.example_input_array = torch.randn(1, 3, self.input_size, self.input_size)

        self.encoder = encoder()
        self.decoder = decoder()

        self.fc1 = nn.Linear(256 * 4 * 4, h_dim)
        self.fc_location = nn.Linear(h_dim, z_dim)
        self.fc_scale = nn.Linear(h_dim, 1)
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 256 * 4 * 4)

        with torch.no_grad():
            self.fc_scale.bias.fill_(1.0e3)

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = self.encoder(x)
        # x = x.view(-1, 256 * 4 * 4)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))

        z_location = self.fc_location(x)
        z_location = torch.nn.functional.normalize(z_location, p=2.0, dim=1)
        # SVAE code: the `+ 1` prevent collapsing behaviors
        z_scale = F.softplus(self.fc_scale(x)) + 1

        return z_location, z_scale

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 256, 4, 4)
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
        best_scaled_image, _, _, _ = self.find_best_rotation(batch)
        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(best_scaled_image)

        loss_recon = self.reconstruction_loss(best_scaled_image, recon)
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

    def project(self, images):
        z_location, _ = self.encode(images)
        return z_location

    def reconstruct(self, coordinates):
        return self.decode(coordinates)

    def reconstruction_loss(self, images, reconstructions):
        return torch.sqrt(
            nn.MSELoss(reduction="none")(
                reconstructions.reshape(-1, self.total_input_size),
                images.reshape(-1, self.total_input_size),
            ).mean(dim=1)
        )
