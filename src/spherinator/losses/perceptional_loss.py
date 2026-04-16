import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for sharper reconstructions."""

    def __init__(self, layers=(3, 8, 15), weights=(1.0, 1.0, 1.0)):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in layers:
            self.blocks.append(vgg[prev : layer + 1])
            prev = layer + 1
        self.weights = weights

    def forward(self, x, target):
        # Expand 1-channel tensors to 3 channels (by repeating the channel)
        # before passing them through the VGG16 blocks, without copying data (expand is a view).
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
            target = target.expand(-1, 3, -1, -1)
        loss = 0.0
        for block, w in zip(self.blocks, self.weights):
            x = block(x)
            target = block(target)
            loss += w * nn.functional.mse_loss(x, target)
        return loss
