from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam

from spherinator.distributions import truncated_normal_distribution


class SphereHead(nn.Module):
    """Sphere head: GAP → Linear → L2-normalize for location, Linear → softplus for scale.

    Reads a spatial feature map ``(B, C, H, W)`` and produces a point on
    S^{z_dim-1} plus a concentration scalar for the PowerSpherical distribution.
    """

    def __init__(
        self,
        latent_channels: int,
        z_dim: int,
        fixed_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_location = nn.Linear(latent_channels, z_dim)
        self.fc_scale = nn.Linear(latent_channels, 1)

        if fixed_scale is not None:
            self.fc_scale.weight.data.zero_()
            self.fc_scale.weight.requires_grad = False
            self.fc_scale.bias.data.fill_(fixed_scale)
            self.fc_scale.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.gap(x).flatten(1)  # (B, C)
        z_location = F.normalize(self.fc_location(x), p=2.0, dim=1)
        z_scale = F.softplus(self.fc_scale(x)) + 1
        return z_location, z_scale


class DualHeadVariationalAutoencoder(pl.LightningModule):
    """VAE with a spatial reconstruction path and a separate sphere head.

    Architecture::

        encoder (spatial)
            ├── spatial latent (B, C, H, W) → decoder   [reconstruction]
            └── sphere head: GAP → Linear → L2-norm → S²  [KL + visualization]

    The decoder receives the full spatial latent — reconstruction never passes
    through the sphere bottleneck.  The sphere head provides a PowerSpherical
    embedding for the KL regularisation term and downstream visualization.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_channels: int,
        z_dim: int = 3,
        beta: float = 1.0,
        loss: str = "MSE",
        fixed_scale: Optional[float] = None,
    ) -> None:
        """
        Args:
            encoder: Spatial encoder outputting ``(B, latent_channels, H, W)``.
            decoder: Spatial decoder accepting ``(B, latent_channels, H, W)``.
            latent_channels: Number of channels in the spatial latent tensor
                (must match encoder output and decoder input).
            z_dim: Dimensionality of the sphere embedding.
            beta: Weight for the KL term (beta-VAE).
            loss: Reconstruction loss ``["MSE", "NLL-normal", "NLL-truncated", "KL"]``.
            fixed_scale: If set, freeze the concentration to this value.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder", "fixed_scale"])

        self.encoder = encoder
        self.decoder = decoder
        self.latent_channels = latent_channels
        self.z_dim = z_dim
        self.beta = beta
        self.loss = loss
        self.fixed_scale = fixed_scale

        self.sphere_head = SphereHead(latent_channels, z_dim, fixed_scale)

        self.example_input_array = getattr(self.encoder, "example_input_array", None)

        if loss == "MSE":
            self.reconstruction_loss = nn.MSELoss()
        elif loss not in ["NLL-normal", "NLL-truncated", "KL"]:
            raise ValueError(f"Loss function {loss} not supported")

    def encode(self, x):
        """Return sphere embedding ``(z_location, z_scale)``."""
        spatial = self.encoder(x)
        return self.sphere_head(spatial)

    def decode(self, x):
        """Decode a spatial latent ``(B, C, H, W)``."""
        return self.decoder(x)

    def reparameterize(self, z_location, z_scale):
        q_z = PowerSpherical(z_location, z_scale)
        p_z = HypersphericalUniform(self.z_dim, device=z_location.device)
        return q_z, p_z

    def forward(self, x):
        spatial = self.encoder(x)
        z_location, z_scale = self.sphere_head(spatial)
        q_z, p_z = self.reparameterize(z_location, z_scale.squeeze())
        recon = self.decoder(spatial)
        return (z_location, z_scale), (q_z, p_z), spatial, recon

    def reconstruct(self, x):
        spatial = self.encoder(x)
        return self.decoder(spatial)

    def on_load_checkpoint(self, checkpoint) -> None:
        if self.fixed_scale is not None:
            state_dict = checkpoint["state_dict"]
            weight_key = "sphere_head.fc_scale.weight"
            bias_key = "sphere_head.fc_scale.bias"
            if weight_key in state_dict:
                state_dict[weight_key] = torch.zeros_like(state_dict[weight_key])
            if bias_key in state_dict:
                state_dict[bias_key] = torch.full_like(state_dict[bias_key], self.fixed_scale)

    def _compute_loss(self, batch, training_step: bool = False):
        if self.loss in ["NLL-normal", "NLL-truncated", "KL"]:
            batch, error = batch

        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(batch)

        if self.loss == "MSE":
            loss_recon = self.reconstruction_loss(batch, recon)
        elif self.loss == "NLL-normal":
            loss_recon = -torch.distributions.Normal(batch, error).log_prob(recon).flatten(1).mean(1)
        elif self.loss == "NLL-truncated":
            loss_recon = -torch.log(
                truncated_normal_distribution(recon, mu=batch, sigma=error, a=0.0, b=1.0).flatten(1).mean(1)
            )
        elif self.loss == "KL":
            q = torch.distributions.Normal(recon, error)
            p = torch.distributions.Normal(batch, error)
            loss_recon = torch.distributions.kl.kl_divergence(q, p).flatten(1).mean(1)
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

        loss_KL = self.beta * torch.distributions.kl.kl_divergence(q_z, p_z)

        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        if training_step:
            self.log("loss_recon", loss_recon, prog_bar=True)
            self.log("loss_KL", loss_KL)
            self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
            self.log("mean(z_location)", torch.mean(z_location))
            self.log("mean(z_scale)", torch.mean(z_scale))

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._compute_loss(batch, training_step=True)
        self.log("train_loss", loss, prog_bar=True)
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
