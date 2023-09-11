import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional

from .spherinator_module import SpherinatorModule

import sys
sys.path.append('external/s-vae-pytorch/')
from hyperspherical_vae.distributions import (HypersphericalUniform,
                                              VonMisesFisher)

class RotationalSphericalVariationalAutoencoder(SpherinatorModule):

    def __init__(self, h_dim=256, z_dim=2, distribution='normal'):
        """
        RotationalSphericalVariationalAutoencoder initializer

        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.randn(1, 3, 64, 64)

        self.h_dim, self.z_dim, self.distribution = h_dim, z_dim, distribution

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=2, padding=2)

        self.fc1 = nn.Linear(256*4*4, h_dim)

        self.fc_mean = nn.Linear(h_dim, z_dim)
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplementedError

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
        if self.distribution == 'normal':
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            length = torch.linalg.vector_norm(z_mean, dim=1)+1.e-20
            z_mean = (z_mean.T / length).T
            #z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1.e-6
        else:
            raise NotImplementedError

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
        return x


    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, device=z_mean.device)
        else:
            raise NotImplementedError
        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x = self.decode(z)

        return (z_mean, z_var), (q_z, p_z), z, x


    def training_step(self, batch, batch_idx):
        images = batch["image"]
        rotations = 36
        losses = torch.zeros(images.shape[0], rotations)
        for i in range(rotations):
            x = functional.rotate(images, 360.0 / rotations * i, expand=False)
            x = functional.center_crop(x, [256,256])
            input = functional.resize(x, [64,64], antialias=False)

            _, (q_z, p_z), _, recon = self.forward(input)

            loss_recon = self.reconstruction_loss(input, recon)

            if self.distribution == 'normal':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif self.distribution == 'vmf':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplementedError

            losses[:,i] = loss_recon + loss_KL

        loss = torch.mean(torch.min(losses, dim=1)[0])

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def reconstruction_loss(self, images, reconstructions):
        return torch.sqrt(torch.sum(torch.square(images.reshape(-1,3*64*64)-reconstructions.reshape(-1,3*64*64)), dim=-1))

    def project(self, images):
        z_mean, _ = self.encode(images)
        return z_mean

    def reconstruct(self, coordinates):
        return self.decode(coordinates)