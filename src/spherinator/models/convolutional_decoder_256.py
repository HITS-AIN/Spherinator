import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalDecoder256(pl.LightningModule):
    def __init__(self, h_dim: int = 256):
        """Convolutional decoder for 256x256 images.
        Input: h_dim
        Output: 3x256x256
        H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        """
        super().__init__()

        self.fc = nn.Linear(h_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1
        )  # 8x8 = 6 - 2 + 3 + 1
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=(4, 4), stride=2, padding=1
        )  # 16x16
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1
        )  # 32x32
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=1
        )  # 64x64
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=(4, 4), stride=2, padding=1
        )  # 128x128
        self.deconv6 = nn.ConvTranspose2d(
            in_channels=16, out_channels=3, kernel_size=(4, 4), stride=2, padding=1
        )  # 256x256

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
        return x
