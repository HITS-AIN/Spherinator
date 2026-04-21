import torch
import torch.nn.functional as F


class SSIMLoss(torch.nn.Module):
    """Structural Similarity Loss (SSIM) as a differentiable loss function."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))
        self.window_size = window_size

    def forward(self, x, y):
        (C,) = x.shape[1:2]  # number of channels
        kernel = self.kernel.expand(C, -1, -1, -1)
        mu_x = F.conv2d(x, kernel, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, kernel, padding=self.window_size // 2, groups=C)
        sigma_x2 = F.conv2d(x * x, kernel, padding=self.window_size // 2, groups=C) - mu_x**2
        sigma_y2 = F.conv2d(y * y, kernel, padding=self.window_size // 2, groups=C) - mu_y**2
        sigma_xy = F.conv2d(x * y, kernel, padding=self.window_size // 2, groups=C) - mu_x * mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return 1 - ssim.mean()
