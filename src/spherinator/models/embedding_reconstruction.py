import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam


class EmbeddingReconstruction(pl.LightningModule):
    def __init__(
        self,
        embedding: nn.Module,
        decoder: nn.Module,
    ):
        """EmbeddingReconstruction initializer

        Args:
            embedding (nn.Module): embedding model
            decoder (nn.Module): decoder model
        """
        super().__init__()
        self.save_hyperparameters()
        self.embedding = embedding
        self.decoder = decoder
        self.example_input_array = torch.tensor([0], dtype=torch.int)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, x):
        x = self.embedding(x)
        return self.decoder(x)

    def training_step(self, batch, _):
        inputs, index = batch
        recon = self.forward(index)
        loss = self.reconstruction_loss(inputs, recon).mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)
