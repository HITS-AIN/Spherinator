import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam


class SphereHead(nn.Module):
    """Sphere head: Linear → L2-normalize for location,
                    Linear → softplus for scale.

    Reads a spatial feature map ``(B, input_dim)`` and produces a point on
    S^{z_dim-1} plus a concentration scalar for the PowerSpherical distribution.
    """

    def __init__(
        self,
        input_dim: int | list | tuple,
        z_dim: int,
        max_scale: float | None = None,
    ) -> None:
        super().__init__()
        flat_dim = math.prod(input_dim) if isinstance(input_dim, (list, tuple)) else input_dim
        self.fc_location = nn.Linear(flat_dim, z_dim)
        self.fc_scale = nn.Linear(flat_dim, 1)
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.flatten(x, 1)
        z_location = F.normalize(self.fc_location(x), p=2.0, dim=1)
        z_scale = F.softplus(self.fc_scale(x)) + 1
        if self.max_scale is not None:
            z_scale = z_scale.clamp(max=self.max_scale)
        return z_location, z_scale


class VariationalAutoencoder(pl.LightningModule):
    """Variational Autoencoder with a hyperspherical latent space."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_out_dim: int,
        z_dim: int = 3,
        beta: float = 1.0,
        reconstruction_loss: nn.Module = nn.MSELoss(),
        max_scale: float | None = None,
    ) -> None:
        """
        Args:
            encoder: Encoder module.
            decoder: Decoder module.
            encoder_out_dim: Dimensionality of the encoder output (must match decoder input).
            z_dim: Dimensionality of the sphere embedding.
            beta: Weight for the KL term (beta-VAE).
            reconstruction_loss: Loss function for reconstruction (default: MSE).
        """
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder", "reconstruction_loss"])

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_output_dim = encoder_out_dim
        self.z_dim = z_dim
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.sphere_head = SphereHead(self.encoder_output_dim, z_dim, max_scale=max_scale)

        self.example_input_array = getattr(self.encoder, "example_input_array", None)

    def encode(self, x):
        """Return sphere embedding ``(z_location, z_scale)``."""
        z = self.encoder(x)
        return self.sphere_head(z)

    def decode(self, z):
        """Decode from the latent space."""
        return self.decoder(z)

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

    def reconstruct(self, x):
        z, _ = self.encode(x)
        return self.decode(z)

    def _compute_loss(self, batch, training_step: bool = False):
        if isinstance(batch, (tuple, list)):
            batch_augmented, batch_original = batch
        else:
            batch_augmented = batch_original = batch
        (_, z_scale), (q_z, p_z), _, recon = self.forward(batch_augmented)
        loss_recon = self.reconstruction_loss(batch_original, recon)
        loss_KL = self.beta * torch.distributions.kl.kl_divergence(q_z, p_z)
        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        if training_step:
            self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
            self.log("mean(z_scale)", torch.mean(z_scale))
            self.log("max(z_scale)", torch.max(z_scale))
            self.log("beta", self.beta)

        return loss, loss_recon, loss_KL

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, loss_recon, loss_KL = self._compute_loss(batch, training_step=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_recon", loss_recon, prog_bar=True)
        self.log("train_loss_KL", loss_KL)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, loss_recon, loss_KL = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_recon", loss_recon, prog_bar=True)
        self.log("val_loss_KL", loss_KL)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, loss_recon, loss_KL = self._compute_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_loss_recon", loss_recon, prog_bar=True)
        self.log("test_loss_KL", loss_KL)
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)
