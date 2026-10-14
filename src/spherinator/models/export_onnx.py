import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from urllib.parse import urljoin

import torch
from torch import nn

from .yaml2model import yaml2model

AIBOM_FILENAME = "aibom.spdx.json"
SPDX_VERSION = "3.0.1"


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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_spdx_aibom(
    aibom_path: str,
    aibom_uri: str,
    exported: list[tuple[str, str, str]],
    model_type: str,
    hyperparameters: dict,
    model_version: str,
    supplier: str,
):
    """Writes an SPDX 3.0.1 AI-BOM (JSON-LD) describing the exported ONNX files.

    Args:
        aibom_path (str): Output path of the AI-BOM file.
        aibom_uri (str): Link to the AI-BOM, used to resolve the download location of each model.
        exported (list): Tuples of (name, ONNX file path, description).
        model_type (str): Type of the exported model, e.g. the class name.
        hyperparameters (dict): Key-value pairs listed as hyperparameters of each model.
        model_version (str): Version of the exported models.
        supplier (str): Name of the organization supplying the models.
    """
    namespace = f"https://spdx.org/spdxdocs/spherinator-{uuid.uuid4()}"
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    creation_info = "_:creationinfo"
    supplier_id = f"{namespace}#supplier"
    tool_id = f"{namespace}#tool"

    packages = [
        {
            "type": "ai_AIPackage",
            "spdxId": f"{namespace}#{name}",
            "creationInfo": creation_info,
            "name": f"Spherinator {name}",
            "description": description,
            "software_packageVersion": model_version,
            "software_downloadLocation": urljoin(aibom_uri, os.path.basename(path)),
            "software_primaryPurpose": "model",
            "releaseTime": now,
            "suppliedBy": supplier_id,
            "verifiedUsing": [{"type": "Hash", "algorithm": "sha256", "hashValue": _sha256(path)}],
            "ai_typeOfModel": [model_type],
            "ai_hyperparameter": [
                {"type": "DictionaryEntry", "key": key, "value": str(value)} for key, value in hyperparameters.items()
            ],
            "ai_autonomyType": "no",
        }
        for name, path, description in exported
    ]
    package_ids = [package["spdxId"] for package in packages]

    aibom = {
        "@context": f"https://spdx.org/rdf/{SPDX_VERSION}/spdx-context.jsonld",
        "@graph": [
            {
                "type": "CreationInfo",
                "@id": creation_info,
                "specVersion": SPDX_VERSION,
                "created": now,
                "createdBy": [supplier_id],
                "createdUsing": [tool_id],
            },
            {
                "type": "Organization",
                "spdxId": supplier_id,
                "creationInfo": creation_info,
                "name": supplier,
            },
            {
                "type": "Tool",
                "spdxId": tool_id,
                "creationInfo": creation_info,
                "name": f"spherinator {version('spherinator')}",
            },
            *packages,
            {
                "type": "SpdxDocument",
                "spdxId": f"{namespace}#document",
                "creationInfo": creation_info,
                "name": "Spherinator ONNX AI-BOM",
                "profileConformance": ["core", "software", "ai"],
                "rootElement": package_ids,
                "element": [supplier_id, tool_id, *package_ids],
            },
        ],
    }

    with open(aibom_path, "w", encoding="utf-8") as f:
        json.dump(aibom, f, indent=2, ensure_ascii=False)


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

    Args:
        ckpt_file (str): Path to the checkpoint file.
        model_file (str): Path to the model YAML file.
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

    graphs = [
        ("encoder", _EncoderWrapper(model), input_shape, "Backbone + sphere head → unit-normalised location vector"),
        ("decoder", _DecoderWrapper(model), latent_shape, "Latent vector → reconstructed output"),
        ("reconstruction", _ReconstructionWrapper(model), input_shape, "End-to-end encode → decode pipeline"),
    ]

    exported = []
    for name, wrapper, shape, description in graphs:
        program = torch.onnx.export(
            wrapper,
            torch.randn(shape, device="cpu"),
            dynamic_axes={"input": {0: "batch"}},
            dynamo=True,
            opset_version=opset_version,
        )
        program.optimize()
        program.model.metadata_props["aibom.uri"] = aibom_uri
        program.model.metadata_props["aibom.format"] = f"SPDX-{SPDX_VERSION}"
        path = os.path.join(export_path, f"{name}.onnx")
        program.save(path)
        exported.append((name, path, description))

    _write_spdx_aibom(
        aibom_path=os.path.join(export_path, AIBOM_FILENAME),
        aibom_uri=aibom_uri,
        exported=exported,
        model_type=type(model).__name__,
        hyperparameters={"input_shape": input_shape, "latent_shape": latent_shape, "opset_version": opset_version},
        model_version=model_version,
        supplier=supplier,
    )
