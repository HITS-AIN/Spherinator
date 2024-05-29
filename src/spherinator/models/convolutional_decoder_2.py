import torch
import torch.nn as nn


class ConvolutionalDecoder2(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )  # 512 x 8 x 8
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 512 x 8 x 8
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 512 x 16 x 16
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )  # 256 x 32 x 32
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # 128 x 64 x 64
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )  # 3 x 128 x 128

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        return x
