from typing import Optional

import torch
import torch.nn as nn
from transformers import ViTModel

from .weights_provider import WeightsProvider


class HuggingFaceViTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        output_dim: Optional[int] = None,
        freeze: bool = False,
        weights: Optional[WeightsProvider] = None,
    ) -> None:
        """HuggingFaceViTEncoder initializer

        Args:
            model_name (str): HuggingFace model name. Defaults to "google/vit-base-patch16-224".
            output_dim (Optional[int]): If set, adds a linear projection on top of the CLS token.
                Defaults to None (uses the model's hidden size directly).
            freeze (bool): Whether to freeze the ViT backbone weights. Defaults to False.
            weights (Optional[WeightsProvider]): Weights to load. Defaults to None.
        """
        super().__init__()

        self.vit = ViTModel.from_pretrained(model_name)

        hidden_size = self.vit.config.hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size

        self.projection = nn.Linear(hidden_size, self.output_dim) if self.output_dim != hidden_size else nn.Identity()

        image_size = self.vit.config.image_size
        self.example_input_array = torch.randn(1, 3, image_size, image_size)

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_token)
