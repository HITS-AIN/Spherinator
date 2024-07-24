import torch
import torch.nn as nn


class ConvolutionalEncoder2(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # 128 x 64 x 64
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )  # 256 x 32 x 32
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 512 x 16 x 16
        self.enc4 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 512 x 8 x 8
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )  # 1024 x 4 x 4
        self.enc6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, latent_dim),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        return x
