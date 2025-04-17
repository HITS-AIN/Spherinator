import os

import torch

from spherinator.models import yaml2model


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
        model.variational_encoder,
        torch.randn(input_shape, device="cpu"),
        dynamic_axes={"input": {0: "batch"}},
        dynamo=True,
    )
    onnx.optimize()
    onnx.save(os.path.join(export_path, "encoder.onnx"))

    onnx = torch.onnx.export(
        model.decoder,
        torch.randn(latent_shape, device="cpu"),
        dynamic_axes={"input": {0: "batch"}},
        dynamo=True,
    )
    onnx.optimize()
    onnx.save(os.path.join(export_path, "decoder.onnx"))
