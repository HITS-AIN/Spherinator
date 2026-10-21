import os
import re

import torch
import yaml
from torch import nn

from .aibom import AIBOM_FILENAME, HF_DATASETS_URL, SPDX_VERSION, write_spdx_aibom
from .yaml2model import yaml2model


def _dataset(model_file: str) -> tuple[str, str | None] | None:
    """Returns name and download location of the training dataset in the ``data`` section of a CLI config.

    A ``path`` that doesn't exist locally and looks like ``owner/name`` is a Hugging Face Hub dataset, as in
    :class:`~spherinator.data.DataModule`. Local paths are reduced to their directory name.
    """
    with open(model_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    init_args = (config.get("data") or {}).get("init_args") or {}
    path = init_args.get("path") or init_args.get("data_directory")
    if path is None:
        return None
    if "path" in init_args and not os.path.exists(path) and re.fullmatch(r"[\w.-]+/[\w.-]+", path):
        return path, HF_DATASETS_URL + path
    return os.path.basename(os.path.normpath(path)), None


class _EncoderWrapper(nn.Module):
    """Exports backbone + sphere_head → z_location (deterministic mean)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if getattr(self.model, "is_variational", True):
            z_location, _ = self.model.sphere_head(self.model.encoder(x))
        else:
            z_location = self.model.encoder(x)
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
    opset_version: int = 19,
    aibom_uri: str = AIBOM_FILENAME,
    model_version: str = "1.0.0",
    supplier: str = "Heidelberg Institute for Theoretical Studies (HITS)",
):
    """Exports the encoder, decoder, and reconstruction graph of a Spherinator model to ONNX format.

    Next to the ONNX files, an SPDX 3.0.1 AI-BOM (``aibom.spdx.json``) is written, which contains the
    SHA-256 of each ONNX file. Each ONNX file links to the AI-BOM via the ``aibom.uri`` metadata property.
    The training dataset (``path`` or ``data_directory``) of the ``data`` section in ``model_file`` is added
    to the AI-BOM as well.

    Args:
        ckpt_file (str): Path to the checkpoint file.
        model_file (str): Path to the model YAML file, e.g. the CLI config of the training run.
        export_path (str): Directory to save the ONNX files.
        input_shape (tuple): Shape of the input tensor.
        latent_shape (tuple): Shape of the latent tensor.
        opset_version (int, optional): ONNX opset version. Defaults to 19.
        aibom_uri (str, optional): Link to the AI-BOM stored in the ONNX metadata. Can be a path relative
            to the ONNX files or an absolute URL. Defaults to "aibom.spdx.json".
        model_version (str, optional): Version of the exported models in the AI-BOM. Defaults to "1.0.0".
        supplier (str, optional): Organization supplying the models in the AI-BOM.
            Defaults to "Heidelberg Institute for Theoretical Studies (HITS)".
    """
    model = yaml2model(model_file)
    checkpoint = torch.load(ckpt_file, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Set dimensions for lazy initialization
    input = torch.randn(input_shape)
    model(input)

    os.makedirs(export_path, exist_ok=False)

    batch = torch.export.Dim("batch")

    # (name, wrapper, example shape, forward argument name, output name, description)
    graphs = [
        (
            "encoder",
            _EncoderWrapper(model),
            input_shape,
            "x",
            "embedding",
            "Backbone + sphere head → unit-normalised location vector",
        ),
        (
            "decoder",
            _DecoderWrapper(model),
            latent_shape,
            "z",
            "reconstruction",
            "Latent vector → reconstructed output",
        ),
        (
            "reconstruction",
            _ReconstructionWrapper(model),
            input_shape,
            "x",
            "reconstruction",
            "End-to-end encode → decode pipeline",
        ),
    ]

    exported = []
    for name, wrapper, shape, arg_name, output_name, description in graphs:
        program = torch.onnx.export(
            wrapper.eval(),
            torch.randn(shape, device="cpu"),
            input_names=["input"],
            output_names=[output_name],
            dynamic_shapes={arg_name: {0: batch}},
            dynamo=True,
            opset_version=opset_version,
        )
        program.optimize()
        program.model.metadata_props["aibom.uri"] = aibom_uri
        program.model.metadata_props["aibom.format"] = f"SPDX-{SPDX_VERSION}"
        path = os.path.join(export_path, f"{name}.onnx")
        program.save(path)
        exported.append((name, path, description))

    write_spdx_aibom(
        aibom_path=os.path.join(export_path, AIBOM_FILENAME),
        aibom_uri=aibom_uri,
        exported=exported,
        model_type=type(model).__name__,
        hyperparameters={"input_shape": input_shape, "latent_shape": latent_shape, "opset_version": opset_version},
        model_version=model_version,
        supplier=supplier,
        dataset=_dataset(model_file),
    )
