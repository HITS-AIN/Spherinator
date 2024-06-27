import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalEncoder(pl.LightningModule):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.conv0 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1
        )  # 128x128
        self.pool0 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 64x64
        self.conv1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )  # 64x64
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 32x32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )  # 32x32
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 16x16
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
        )  # 16x16
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 8x8
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1
        )  # 8x8
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)  # 4x4

        self.fc1 = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x
