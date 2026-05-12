import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam


class SphereHead(nn.Module):
    """Sphere head: GAP → Linear → L2-normalize for location, Linear → softplus for scale.

    Reads a spatial feature map ``(B, C, H, W)`` and produces a point on
    S^{z_dim-1} plus a concentration scalar for the PowerSpherical distribution.
    """

    def __init__(
        self,
        latent_channels: int,
        z_dim: int,
    ) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_location = nn.Linear(latent_channels, z_dim)
        self.fc_scale = nn.Linear(latent_channels, 1)

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
        reconstruction_loss: nn.Module = nn.MSELoss(),
    ) -> None:
        """
        Args:
            encoder: Spatial encoder outputting ``(B, latent_channels, H, W)``.
            decoder: Spatial decoder accepting ``(B, latent_channels, H, W)``.
            latent_channels: Number of channels in the spatial latent tensor
                (must match encoder output and decoder input).
            z_dim: Dimensionality of the sphere embedding.
            beta: Weight for the KL term (beta-VAE).
            reconstruction_loss: Loss function for reconstruction (default: MSE).
        """
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.latent_channels = latent_channels
        self.z_dim = z_dim
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.sphere_head = SphereHead(latent_channels, z_dim)

        self.example_input_array = getattr(self.encoder, "example_input_array", None)

    # def encode(self, x):
    #     """Return sphere embedding ``(z_location, z_scale)``."""
    #     spatial = self.encoder(x)
    #     return self.sphere_head(spatial)

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

    def _compute_loss(self, batch, training_step: bool = False):
        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(batch)
        loss_recon = self.reconstruction_loss(batch, recon)
        loss_KL = self.beta * torch.distributions.kl.kl_divergence(q_z, p_z)
        loss = (loss_recon + loss_KL).mean()
        loss_recon = loss_recon.mean()
        loss_KL = loss_KL.mean()

        if training_step:
            self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
            self.log("mean(z_location)", torch.mean(z_location))
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
