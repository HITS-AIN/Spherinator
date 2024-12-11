from typing import Optional

import lightning.pytorch as pl
import torch
import torch.linalg
import torch.nn as nn

from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        h_dim: int = 256,
        z_dim: int = 3,
    ):
        """Autoencoder initializer

        Args:
            encoder (Optional[nn.Module], optional): encoder model. Defaults to None.
            decoder (Optional[nn.Module], optional): decoder model. Defaults to None.
            h_dim (int, optional): dimension of the hidden layers. Defaults to 256.
            z_dim (int, optional): dimension of the latent representation. Defaults to 3.
        """
        super().__init__()

        if encoder is None:
            encoder = ConvolutionalEncoder(latent_dim=h_dim)
        if decoder is None:
            decoder = ConvolutionalDecoder(latent_dim=h_dim)

        # self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.example_input_array = self.encoder.example_input_array
        # self.example_input_array = torch.randn(2, 1, 12)

        self.fc = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)

        self.reconstruction_loss = nn.MSELoss()
        # self.reconstruction_loss = nn.CrossEntropyLoss()

        with torch.no_grad():
            self.fc_scale.bias.fill_(1.0e3)

    def encode(self, x):
        x = self.encoder(x)
        z = self.fc(x)
        return z

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = self.decoder(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def training_step(self, batch, batch_idx):
        recon = self.forward(batch)
        loss = self.reconstruction_loss(batch, recon).mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss
