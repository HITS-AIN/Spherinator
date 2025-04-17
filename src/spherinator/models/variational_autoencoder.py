from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam

from spherinator.distributions import truncated_normal_distribution


class VariationalEncoder(nn.Module):
    """Variational encoder for extra layer splitting the location and scale of the latent space."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_out_dim: int,
        z_dim: int,
        fixed_scale: Optional[float] = None,
    ) -> None:
        """VariationalEncoder initializer

        Args:
            encoder (nn.Module): encoder model
            encoder_out_dim (int): output dimension of the encoder
            z_dim (int): latent space dimension
            fixed_scale (Optional[float], optional): fixed scale value for the latent space. Defaults to None.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_out_dim = encoder_out_dim
        self.z_dim = z_dim

        self.fc_location = nn.Linear(self.encoder_out_dim, self.z_dim)
        self.fc_scale = nn.Linear(self.encoder_out_dim, 1)

        # Set scale output and freeze the parameter
        if fixed_scale is not None:
            self.fc_scale.weight.data.zero_()
            self.fc_scale.weight.requires_grad = False
            self.fc_scale.bias.data.fill_(fixed_scale)
            self.fc_scale.bias.requires_grad = False

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        z_location = self.fc_location(x)
        z_location = torch.nn.functional.normalize(z_location, p=2.0, dim=1)
        # SVAE code: the `+ 1` prevent collapsing behaviors
        z_scale = F.softplus(self.fc_scale(x)) + 1
        return z_location, z_scale


class VariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_out_dim: int,
        z_dim: int = 3,
        beta: float = 1.0,
        loss: str = "MSE",
        fixed_scale: Optional[float] = None,
    ) -> None:
        """VariationalAutoencoder initializer

        Args:
            encoder (nn.Module): encoder model
            decoder (nn.Module): decoder model
            encoder_out_dim (int): output dimension of the encoder
            z_dim (int, optional): latent space dimension. Defaults to 3.
            beta (float, optional): factor for beta-VAE. Defaults to 1.0.
            loss (str, optional): loss function ["MSE", "NLL-normal", "NLL-truncated", "KL"].
                                  Defaults to "MSE".
            fixed_scale (Optional[float], optional): fixed scale value for the latent space. Defaults to None.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder"])
        # self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_out_dim = encoder_out_dim
        self.z_dim = z_dim
        self.beta = beta
        self.loss = loss

        self.variational_encoder = VariationalEncoder(
            encoder, self.encoder_out_dim, self.z_dim, fixed_scale
        )

        self.example_input_array = getattr(self.encoder, "example_input_array", None)

        if loss == "MSE":
            self.reconstruction_loss = nn.MSELoss()
        elif loss not in ["NLL-normal", "NLL-truncated", "KL"]:
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

    def pure_forward(self, x):
        z_location, _ = self.encode(x)
        return self.decode(z_location)

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        if self.loss in ["NLL-normal", "NLL-truncated", "KL"]:
            batch, error = batch

        (z_location, z_scale), (q_z, p_z), _, recon = self.forward(batch)

        if self.loss == "MSE":
            loss_recon = self.reconstruction_loss(batch, recon)
        elif self.loss == "NLL-normal":
            loss_recon = (
                -torch.distributions.Normal(batch, error)
                .log_prob(recon)
                .flatten(1)
                .mean(1)
            )
        elif self.loss == "NLL-truncated":
            loss_recon = -torch.log(
                truncated_normal_distribution(
                    recon, mu=batch, sigma=error, a=0.0, b=1.0
                )
                .flatten(1)
                .mean(1)
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
