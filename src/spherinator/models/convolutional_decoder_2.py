import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalDecoder2(nn.Module):
    def __init__(self, h_dim: int = 256):
        super().__init__()

        self.fc = nn.Linear(h_dim, 256 * 8 * 8)
        self.deconv0 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
        )
        self.upsample0 = nn.Upsample(scale_factor=2)  # 16x16
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )
        self.upsample1 = nn.Upsample(scale_factor=2)  # 32x32
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )
        self.upsample2 = nn.Upsample(scale_factor=2)  # 64x64
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=(3, 3), stride=1, padding=1
        )
        self.upsample3 = nn.Upsample(scale_factor=2)  # 128x128

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc(x))
        x = x.view(-1, 256, 8, 8)
        x = F.relu(self.deconv0(x))
        x = self.upsample0(x)
        x = F.relu(self.deconv1(x))
        x = self.upsample1(x)
        x = F.relu(self.deconv2(x))
        x = self.upsample2(x)
        x = self.deconv3(x)
        x = self.upsample3(x)
        return x
