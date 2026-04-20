from typing import Optional

import torch
import torch.nn as nn
from transformers import ResNetModel

from .weights_provider import WeightsProvider


class HuggingFaceResNetEncoder(nn.Module):
    """HuggingFaceResNetEncoder is a PyTorch module that wraps a Hugging Face ResNetModel to be
    used as an encoder in the Spherinator framework. It takes an image input and produces a
    fixed-size embedding vector. The pooled output from the ResNet model is used as the output
    representation, and an optional linear projection can be applied on top of it to match a
    desired output dimension.

    Args:
        model_name (str): HuggingFace model name. Defaults to "microsoft/resnet-18".
        output_dim (Optional[int]): If set, adds a linear projection on top of the pooled output.
            Defaults to None (uses the model's hidden size directly).
        freeze (bool): Whether to freeze the ResNet backbone weights. Defaults to False.
        weights (Optional[WeightsProvider]): Weights to load. Defaults to None.
    """

    def __init__(
        self,
        model_name: str = "microsoft/resnet-18",
        output_dim: Optional[int] = None,
        freeze: bool = False,
        weights: Optional[WeightsProvider] = None,
    ) -> None:
        super().__init__()

        self.resnet = ResNetModel.from_pretrained(model_name)

        hidden_size = self.resnet.config.hidden_sizes[-1]
        self.output_dim = output_dim if output_dim is not None else hidden_size

        self.projection = nn.Linear(hidden_size, self.output_dim) if self.output_dim != hidden_size else nn.Identity()

        image_size = 128
        self.example_input_array = torch.randn(1, 3, image_size, image_size)

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.resnet(pixel_values=x)
        pooled = outputs.pooler_output.flatten(1)
        return self.projection(pooled)
