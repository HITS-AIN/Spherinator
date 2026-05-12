import os

import torch
import torch.nn as nn

from .yaml2model import yaml2model


class _EncoderWrapper(nn.Module):
    """Exports backbone + sphere_head → z_location (deterministic mean)."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.sphere_head = model.sphere_head
        self.model = model

    def forward(self, x):
        if getattr(self.model, "is_variational", True):
            z_location, _ = self.sphere_head(self.encoder(x))
        else:
            z_location = self.encoder(x)
        return z_location


class _DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder

    def forward(self, z):
        return self.decoder(z)


class _ReconstructionWrapper(nn.Module):
    """Wraps a model's reconstruct() method as a plain nn.Module for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, x):
        return self._model.reconstruct(x)


def export_onnx(
    ckpt_file: str,
    model_file: str,
    export_path: str,
    input_shape: tuple,
    latent_shape: tuple,
):
    model = yaml2model(model_file)
    checkpoint = torch.load(ckpt_file, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Set dimensions for lazy initialization
    input = torch.randn(input_shape)
    model(input)

    os.makedirs(export_path, exist_ok=False)

    onnx = torch.onnx.export(
        _EncoderWrapper(model),
        torch.randn(input_shape, device="cpu"),
        dynamic_axes={"input": {0: "batch"}},
        dynamo=True,
    )
    onnx.optimize()
    onnx.save(os.path.join(export_path, "encoder.onnx"))

    onnx = torch.onnx.export(
        _DecoderWrapper(model),
        torch.randn(latent_shape, device="cpu"),
        dynamic_axes={"input": {0: "batch"}},
        dynamo=True,
    )
    onnx.optimize()
    onnx.save(os.path.join(export_path, "decoder.onnx"))

    onnx = torch.onnx.export(
        _ReconstructionWrapper(model),
        torch.randn(input_shape, device="cpu"),
        dynamic_axes={"input": {0: "batch"}},
        dynamo=True,
    )
    onnx.optimize()
    onnx.save(os.path.join(export_path, "reconstruction.onnx"))
