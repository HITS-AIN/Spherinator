from typing import Any, Mapping, Optional

import torch


class WeightsProvider:
    """Class to load and provide weights for the model."""

    def __init__(self, weight_path, prefix: Optional[str] = None) -> None:
        """WeightsProvider initializer
        Args:
            weight_path (str): The path to the weights file
            prefix (Optional[str], optional): The prefix to use when loading the weights. Defaults to None.
        """
        self.weights = torch.load(weight_path)["state_dict"]
        if prefix is not None:
            self.weights = {k[len(prefix) + 1 :]: v for k, v in self.weights.items() if k.startswith(prefix)}

    def get_state_dict(self) -> Mapping[str, Any]:
        """Get the state dict of the model."""
        return self.weights
