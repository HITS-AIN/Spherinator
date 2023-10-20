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
sys.path.append(os.path.join(script_dir, '../external/power_spherical/'))
from power_spherical import (HypersphericalUniform, PowerSpherical)


class RotationalVariationalAutoencoderPower(SpherinatorModule):

    def __init__(self,
                 h_dim: int = 256,
                 z_dim: int = 2,
                 image_size: int = 91,
                 rotations: int = 36,
                 beta: float = 1.0):
        """
        RotationalVariationalAutoencoder initializer

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
        self.input_size = 64

        if self.input_size > self.crop_size:
            raise ValueError("Image size to small.")

        self.example_input_array = torch.randn(1, 3, self.input_size, self.input_size)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=2, padding=2)

        self.fc1 = nn.Linear(256*4*4, h_dim)
        self.fc_mean = nn.Linear(h_dim, z_dim)
        self.fc_var = nn.Linear(h_dim, 1)
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 256*4*4)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4,4), stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(5,5), stride=1, padding=2)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 256*4*4)
        x = F.tanh(self.fc1(x))

        z_mean = self.fc_mean(x)
        z_mean = torch.nn.functional.normalize(z_mean, p=2.0, dim=1)
        # SVAE code: the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.fc_var(x)) + 1

        return z_mean, z_var

    def decode(self, z):
        x = F.tanh(self.fc2(z))
        x = F.tanh(self.fc3(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x

    def reparameterize(self, z_mean, z_var):
        q_z = PowerSpherical(z_mean, z_var)
        p_z = HypersphericalUniform(self.z_dim - 1, device=z_mean.device)
        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var.squeeze())
        z = q_z.rsample()
        recon = self.decode(z)
        return (z_mean, z_var), (q_z, p_z), z, recon

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        losses = torch.zeros(images.shape[0], self.rotations)
        losses_recon = torch.zeros(images.shape[0], self.rotations)
        losses_KL = torch.zeros(images.shape[0], self.rotations)
        for i in range(self.rotations):
            rotate = functional.rotate(images, 360.0 / self.rotations * i, expand=False)
            crop = functional.center_crop(rotate, [self.crop_size, self.crop_size])
            scaled = functional.resize(crop, [self.input_size, self.input_size], antialias=False)

            (_, _), (q_z, p_z), _, recon = self.forward(scaled)

            loss_recon = self.reconstruction_loss(scaled, recon)
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

            losses[:,i] = loss_recon + self.beta * loss_KL
            losses_recon[:,i] = loss_recon
            losses_KL[:,i] = loss_KL

        loss_idx = torch.min(losses, dim=1)[1]
        loss = torch.mean(torch.gather(losses, 1, loss_idx.unsqueeze(1)))
        loss_recon = torch.mean(torch.gather(losses_recon, 1, loss_idx.unsqueeze(1)))
        loss_KL = torch.mean(torch.gather(losses_KL, 1, loss_idx.unsqueeze(1)))
        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_recon', loss_recon, prog_bar=True)
        self.log('loss_KL', loss_KL)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
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
        return nn.MSELoss(reduction='none')(
            reconstructions.reshape(-1, 3*64*64), images.reshape(-1, 3*64*64)).sum(-1).mean()
