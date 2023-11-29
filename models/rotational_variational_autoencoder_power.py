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
sys.path.append(os.path.join(script_dir, "../external/power_spherical/"))
from power_spherical import HypersphericalUniform, PowerSpherical


class RotationalVariationalAutoencoderPower(SpherinatorModule):
    def __init__(
        self,
        h_dim: int = 256,
        z_dim: int = 2,
        image_size: int = 91,
        rotations: int = 36,
        beta: float = 1.0,
    ):
        """
        RotationalVariationalAutoencoderPower initializer

        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param image_size: size of the input images
        :param rotations: number of rotations
        :param beta: factor for beta-VAE
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

        self.conv0 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1
        )  # 128x128
        self.pool0 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 64x64
        self.conv1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )  # 64x64
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 32x32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )  # 32x32
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 16x16
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
        )  # 16x16
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 8x8
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1
        )  # 8x8
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 4x4

        self.fc1 = nn.Linear(256 * 4 * 4, h_dim)
        self.fc_location = nn.Linear(h_dim, z_dim)
        self.fc_scale = nn.Linear(h_dim, 1)
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 256 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1
        )  # 8x8
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=(4, 4), stride=2, padding=1
        )  # 16x16
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
        )  # 32x32
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=1
        )  # 64x64
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1
        )  # 127x127
        self.deconv6 = nn.ConvTranspose2d(
            in_channels=16, out_channels=3, kernel_size=(2, 2), stride=1, padding=0
        )  # 128x128

        with torch.no_grad():
            self.fc_scale.bias.fill_(1.0e3)

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, 256 * 4 * 4)
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

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
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
        data, _ = batch
        best_recon = torch.ones(data.shape[0], device=data.device) * 1e10
        best_scaled = torch.zeros(
            (data.shape[0], data.shape[1], self.input_size, self.input_size),
            device=data.device,
        )

        with torch.no_grad():
            for i in range(self.rotations):
                rotate = functional.rotate(
                    data, 360.0 / self.rotations * i, expand=False
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

        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(best_scaled)

        loss_recon = self.reconstruction_loss(best_scaled, recon)
        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z) * self.beta
        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_recon", loss_recon, prog_bar=True)
        self.log("loss_KL", loss_KL)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        self.log("mean(z_location) ", torch.mean(z_location))
        self.log("mean(z_scale) ", torch.mean(z_scale))
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
