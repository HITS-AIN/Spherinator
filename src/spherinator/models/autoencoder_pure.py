import lightning.pytorch as pl
import torch.nn as nn
from torch.optim import Adam


class AutoencoderPure(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        """Autoencoder initializer

        Args:
            encoder (nn.Module): encoder model
            decoder (nn.Module): decoder model
        """
        super().__init__()

        # self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder

        self.example_input_array = self.encoder.example_input_array
        # self.example_input_array = torch.randn(2, 1, 12)

        self.reconstruction_loss = nn.MSELoss()
        # self.reconstruction_loss = nn.CrossEntropyLoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

    def training_step(self, batch, batch_idx):
        recon = self.forward(batch)
        loss = self.reconstruction_loss(batch, recon).mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)
