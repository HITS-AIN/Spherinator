import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from .truncated_normal_distribution import truncated_normal_distribution


class AutoencoderPure(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss: str = "MSE",
    ):
        """Autoencoder initializer

        Args:
            encoder (nn.Module): encoder model
            decoder (nn.Module): decoder model
            loss (str, optional): loss function ["MSE", "NLL"]. Defaults to "MSE".
        """
        super().__init__()

        # self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss

        self.example_input_array = self.encoder.example_input_array
        # self.example_input_array = torch.randn(2, 1, 12)

        if loss == "MSE":
            self.reconstruction_loss = nn.MSELoss()
        elif loss != "NLL":
            raise ValueError(f"Loss function {loss} not supported")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        if self.loss == "NLL":
            batch, error = batch

        recon = self.forward(batch)

        if self.loss == "MSE":
            loss = self.reconstruction_loss(batch, recon).mean()
        elif self.loss == "NLL":
            loss = -torch.log(
                truncated_normal_distribution(
                    recon, mu=batch, sigma=error, a=0.0, b=1.0
                ).mean()
            )

        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)
