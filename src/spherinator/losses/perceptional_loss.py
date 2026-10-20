import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for sharper reconstructions."""

    def __init__(self, layers=(3, 8, 15), weights=(1.0, 1.0, 1.0)):
        super().__init__()
        if len(layers) != len(weights):
            raise ValueError("layers and weights must have the same length")
        if not layers or any(layer < 0 for layer in layers) or tuple(layers) != tuple(sorted(layers)):
            raise ValueError("layers must be a non-empty, sorted sequence of non-negative indices")
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        if layers[-1] >= len(vgg):
            raise ValueError(f"layer index {layers[-1]} is outside VGG16 features")
        for p in vgg.parameters():
            p.requires_grad = False
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.blocks.append(vgg[prev : layer + 1])
            prev = layer + 1
        self.weights = weights
        # Non-persistent to keep checkpoints compatible with versions without these buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x, target):
        # Expand 1-channel tensors to 3 channels (by repeating the channel)
        # before passing them through the VGG16 blocks, without copying data (expand is a view).
        if x.ndim != 4 or target.shape != x.shape or x.shape[1] not in (1, 3):
            raise ValueError("x and target must have matching NCHW shapes with 1 or 3 channels")
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
            target = target.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for block, w in zip(self.blocks, self.weights):
            x = block(x)
            target = block(target)
            loss += w * nn.functional.mse_loss(x, target)
        return loss
