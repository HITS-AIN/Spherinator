import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import sys
sys.path.append('external/s-vae-pytorch/')
from hyperspherical_vae.distributions import (HypersphericalUniform,
                                              VonMisesFisher)


class SVAE(pl.LightningModule):

    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        SVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super().__init__()

        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution

        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplementedError

        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplementedError

        return z_mean, z_var

    def decode(self, z):

        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)

        return x

    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplementedError

        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)

        return (z_mean, z_var), (q_z, p_z), z, x_


    def training_step(self, batch, batch_idx):

        x, _ = batch

        # dynamic binarization
        x = (x > torch.distributions.Uniform(0, 1).sample(x.shape)).float()

        _, (q_z, p_z), _, x_ = self.forward(x)

        loss_recon = nn.BCEWithLogitsLoss(reduction='none')(
            x_, x.reshape(-1, 784)).sum(-1).mean()

        if self.distribution == 'normal':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        elif self.distribution == 'vmf':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            raise NotImplementedError

        loss = loss_recon + loss_KL
        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
