import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss: nn.Module = nn.MSELoss(),
    ):
        """Autoencoder initializer

        Args:
            encoder (nn.Module): encoder model
            decoder (nn.Module): decoder model
            loss (nn.Module, optional): loss function. Defaults to nn.MSELoss().
        """
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss

        self.example_input_array = getattr(self.encoder, "example_input_array", None)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    def reconstruct(self, x):
        return self.forward(x)

    def _compute_loss(self, batch):
        recon = self.forward(batch)
        return self.loss(batch, recon).mean()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)
