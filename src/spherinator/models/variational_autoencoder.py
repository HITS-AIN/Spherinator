import math
from typing import List, Optional

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
        input_dim: int,
        z_dim: int,
        fixed_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.fc_location = nn.Linear(input_dim, z_dim)
        self.fc_scale = nn.Linear(input_dim, 1)
        self.fixed_scale = fixed_scale

        if fixed_scale is not None:
            self.fc_scale.weight.data.zero_()
            self.fc_scale.weight.requires_grad = False
            self.fc_scale.bias.data.fill_(fixed_scale)
            self.fc_scale.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_location = F.normalize(self.fc_location(x), p=2.0, dim=1)
        z_scale = F.softplus(self.fc_scale(x)) + 1
        return z_location, z_scale


class VariationalAutoencoder(pl.LightningModule):
    """Variational Autoencoder with a hyperspherical latent space."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_out_dim: List[int],
        z_dim: int = 3,
        beta: float = 1.0,
        reconstruction_loss: nn.Module = nn.MSELoss(),
        fixed_scale: Optional[float] = None,
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

        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_out_dim = (encoder_out_dim,) if isinstance(encoder_out_dim, int) else tuple(encoder_out_dim)
        self.encoder_output_flat_dim = math.prod(self.encoder_out_dim)
        self.z_dim = z_dim
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss

        self.sphere_head = SphereHead(self.encoder_output_flat_dim, z_dim, fixed_scale=fixed_scale)
        self.fc_decode = nn.Linear(z_dim, self.encoder_output_flat_dim)

        self.example_input_array = getattr(self.encoder, "example_input_array", None)

    def encode(self, x):
        """Return sphere embedding ``(z_location, z_scale)``."""
        spatial = self.encoder(x)
        flat = spatial.flatten(1)
        return self.sphere_head(flat)

    def decode(self, x):
        """Decode a spatial latent ``(B, C, H, W)``."""
        return self.decoder(x)

    def reparameterize(self, z_location, z_scale):
        q_z = PowerSpherical(z_location, z_scale)
        p_z = HypersphericalUniform(self.z_dim, device=z_location.device)
        return q_z, p_z

    def forward(self, x):
        z_location, z_scale = self.encode(x)
        q_z, p_z = self.reparameterize(z_location, z_scale.squeeze())
        z = q_z.rsample()
        if len(self.encoder_out_dim) > 1:
            z_spatial = self.fc_decode(z).view(z.shape[0], *self.encoder_out_dim)
        else:
            z_spatial = z
        recon = self.decoder(z_spatial)
        return (z_location, z_scale), (q_z, p_z), z, recon

    def reconstruct(self, x):
        spatial = self.encoder(x)
        return self.decoder(spatial)

    def _compute_loss(self, batch, training_step: bool = False):
        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(batch)
        loss_recon = self.reconstruction_loss(batch, recon)
        loss_KL = self.beta * torch.distributions.kl.kl_divergence(q_z, p_z)
        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        if training_step:
            self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
            self.log("mean(z_scale)", torch.mean(z_scale))

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
