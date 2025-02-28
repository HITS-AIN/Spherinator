import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam

from .truncated_normal_distribution import truncated_normal_distribution


class VariationalEncoder(nn.Module):
    """Variational encoder for extra layer splitting the location and scale of the latent space."""

    def __init__(
        self,
        encoder: nn.Module,
        z_dim: int,
    ) -> None:
        """VariationalEncoder initializer

        Args:
            encoder (nn.Module): encoder model
            z_dim (int): latent space dimension
        """
        super().__init__()
        self.encoder = encoder
        self.z_dim = z_dim

        self.fc_location = nn.LazyLinear(self.z_dim)
        self.fc_scale = nn.LazyLinear(1)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        z_location = self.fc_location(x)
        z_location = torch.nn.functional.normalize(z_location, p=2.0, dim=1)
        # SVAE code: the `+ 1` prevent collapsing behaviors
        z_scale = F.softplus(self.fc_scale(x)) + 1
        return z_location, z_scale


class VariationalAutoencoderPure(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int = 3,
        beta: float = 1.0,
        loss: str = "MSE",
    ) -> None:
        """Autoencoder initializer

        Args:
            encoder (nn.Module): encoder model
            decoder (nn.Module): decoder model
            z_dim (int, optional): latent space dimension. Defaults to 3.
            beta (float, optional): factor for beta-VAE. Defaults to 1.0.
            loss (str, optional): loss function ["MSE", "NLL"]. Defaults to "MSE".
        """
        super().__init__()

        # self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = z_dim
        self.beta = beta
        self.loss = loss

        self.variational_encoder = VariationalEncoder(encoder, self.z_dim)

        self.example_input_array = self.encoder.example_input_array
        # self.example_input_array = torch.randn(2, 1, 12)

        if loss == "MSE":
            self.reconstruction_loss = nn.MSELoss()
        elif loss != "NLL":
            raise ValueError(f"Loss function {loss} not supported")

    def encode(self, x):
        return self.variational_encoder(x)

    def decode(self, x):
        return self.decoder(x)

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

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        if self.loss == "NLL":
            batch, error = batch

        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(batch)

        if self.loss == "MSE":
            loss_recon = self.reconstruction_loss(batch, recon)
        elif self.loss == "NLL":
            loss_recon = -torch.log(
                truncated_normal_distribution(
                    recon, mu=batch, sigma=error, a=0.0, b=1.0
                ).mean(0)
            )

        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z)

        loss = (loss_recon + self.beta * loss_KL).mean()
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
