import inspect
import os
import re

import onnx
import torch
import yaml
from torch import nn

from .aibom import AIBOM_FILENAME, HF_DATASETS_URL, SPDX_VERSION, write_spdx_aibom
from .yaml2model import yaml2model


def _dataset(config: dict) -> tuple[str, str | None] | None:
    """Returns name and download location of the training dataset in the ``data`` section of a CLI config.

    A ``path`` that doesn't exist locally and looks like ``owner/name`` is a Hugging Face Hub dataset, as in
    :class:`~spherinator.data.DataModule`. Parquet files and explicit relative paths (``./``, ``../``) are
    always local, even if they don't exist on the export machine. Local paths are reduced to their last component.
    """
    init_args = (config.get("data") or {}).get("init_args") or {}
    path = init_args.get("path") or init_args.get("data_directory")
    if path is None:
        return None
    if (
        "path" in init_args
        and not os.path.exists(path)
        and not path.endswith(".parquet")
        and not path.startswith(("./", "../"))
        and re.fullmatch(r"[\w.-]+/[\w.-]+", path)
    ):
        return path, HF_DATASETS_URL + path
    return os.path.basename(os.path.normpath(path)), None


def _hyperparameters(model: nn.Module, init_args: dict) -> dict:
    """Returns the hyperparameters of a model, i.e. the scalar arguments of its constructor like ``z_dim`` or ``beta``.

    Values in ``init_args`` of the config take precedence over the defaults of the constructor. Submodules like
    encoder and decoder, and unset values (None) are skipped.
    """
    parameters = inspect.signature(type(model).__init__).parameters.values()
    values = {p.name: p.default for p in parameters if p.default is not inspect.Parameter.empty}
    values.update(init_args)

    def is_scalar(value) -> bool:
        return isinstance(value, (bool, int, float, str))

    return {
        key: value
        for key, value in values.items()
        if is_scalar(value) or (isinstance(value, (list, tuple)) and value and all(map(is_scalar, value)))
    }


def _signature(path: str, inputs: dict[str, str], outputs: dict[str, str]) -> str:
    """Describes the ONNX opset and the inputs and outputs of a saved ONNX file with their data types and shapes.

    Args:
        path (str): Path of the ONNX file.
        inputs (dict): Expected input names, in order, mapped to their meaning.
        outputs (dict): Expected output names, in order, mapped to their meaning.

    Returns:
        str: Description like ``ONNX opset 19. Inputs: input float32[batch, 1, 12] (input data). Outputs: …``.

    Raises:
        RuntimeError: If the input or output names of the file differ from the expected ones.
    """
    model = onnx.load(path, load_external_data=False)
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    graph = model.graph

    def describe(kind: str, values, expected: dict[str, str]) -> str:
        names = [value.name for value in values]
        if names != list(expected):
            raise RuntimeError(f"{os.path.basename(path)} has {kind} {names}, expected {list(expected)}")
        descriptions = []
        for value in values:
            tensor = value.type.tensor_type
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type).name
            shape = ", ".join(dim.dim_param or str(dim.dim_value) for dim in tensor.shape.dim)
            descriptions.append(f"{value.name} {dtype}[{shape}] ({expected[value.name]})")
        return f"{kind.capitalize()}: {', '.join(descriptions)}."

    return f"ONNX opset {opset}. {describe('inputs', graph.input, inputs)} {describe('outputs', graph.output, outputs)}"


class _EncoderWrapper(nn.Module):
    """Exports backbone + sphere_head → (z_location, concentration) for variational models, otherwise the
    encoder output."""

    def __init__(self, model, variational: bool):
        super().__init__()
        self.model = model
        self.variational = variational

    def forward(self, x):
        if self.variational:
            z_location, z_scale = self.model.sphere_head(self.model.encoder(x))
            return z_location, z_scale.squeeze(-1)
        return self.model.encoder(x)


class _DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder

    def forward(self, z):
        return self.decoder(z)


class _ReconstructionWrapper(nn.Module):
    """Exports the end-to-end encode → decode pipeline. Variational models decode the location vector and
    additionally return the concentration."""

    def __init__(self, model, variational: bool):
        super().__init__()
        self._model = model
        self.variational = variational

    def forward(self, x):
        if self.variational:
            z_location, z_scale = self._model.encode(x)
            return self._model.decode(z_location), z_scale.squeeze(-1)
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

    Outputs of the exported graphs:

    - ``encoder.onnx``: ``embedding`` and, for variational models, ``concentration``
    - ``decoder.onnx``: ``reconstruction``
    - ``reconstruction.onnx``: ``reconstruction`` and, for variational models, ``concentration``

    The ``concentration`` of shape ``(batch,)`` is the scale of the PowerSpherical distribution, i.e. the
    confidence of the embedding. The AI-BOM records the ONNX opset and the inputs and outputs of each graph, with
    their data types and shapes, in ``ai_informationAboutApplication``. Its ``ai_hyperparameter`` entries are the
    scalar arguments of the model constructor, e.g. ``z_dim``, ``beta``, and ``max_scale``.

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

    Raises:
        TypeError: If the model doesn't define ``is_variational``.
        RuntimeError: If a graph doesn't have the expected inputs and outputs, or if the export writes files that the
            AI-BOM doesn't list, e.g. ONNX external data of models larger than 2 GB.
    """
    with open(model_file, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = yaml2model(model_file)
    variational = getattr(model, "is_variational", None)
    if not isinstance(variational, bool):
        raise TypeError(f"{type(model).__name__} doesn't define is_variational and can't be exported to ONNX")
    checkpoint = torch.load(ckpt_file, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Set dimensions for lazy initialization
    input = torch.randn(input_shape)
    model(input)

    os.makedirs(export_path, exist_ok=False)

    batch = torch.export.Dim("batch")
    if variational:
        encoder_description = "Backbone + sphere head → unit-normalised location vector and concentration"
        embedding = "unit-normalised location vector on the hypersphere"
        concentration = {
            "concentration": "scale κ ≥ 1 of the Power Spherical distribution around the embedding; "
            "higher means a more confident embedding"
        }
    else:
        encoder_description = "Backbone → latent vector"
        embedding = "latent vector"
        concentration = {}
    and_concentration = " and concentration" if variational else ""

    # (name, wrapper, example shape, forward argument name, inputs, outputs, description). Inputs and outputs map
    # the names of the graph to their meaning.
    graphs = [
        (
            "encoder",
            _EncoderWrapper(model, variational),
            input_shape,
            "x",
            {"input": "input data"},
            {"embedding": embedding, **concentration},
            encoder_description,
        ),
        (
            "decoder",
            _DecoderWrapper(model),
            latent_shape,
            "z",
            {"input": "latent vector"},
            {"reconstruction": "reconstructed input"},
            "Latent vector → reconstructed output",
        ),
        (
            "reconstruction",
            _ReconstructionWrapper(model, variational),
            input_shape,
            "x",
            {"input": "input data"},
            {"reconstruction": "reconstructed input", **concentration},
            f"End-to-end encode → decode pipeline{and_concentration}",
        ),
    ]

    exported = []
    for name, wrapper, shape, arg_name, inputs, outputs, description in graphs:
        program = torch.onnx.export(
            wrapper.eval(),
            torch.randn(shape, device="cpu"),
            input_names=list(inputs),
            output_names=list(outputs),
            dynamic_shapes={arg_name: {0: batch}},
            dynamo=True,
            opset_version=opset_version,
        )
        program.optimize()
        program.model.metadata_props["aibom.uri"] = aibom_uri
        program.model.metadata_props["aibom.format"] = f"SPDX-{SPDX_VERSION}"
        path = os.path.join(export_path, f"{name}.onnx")
        program.save(path)
        exported.append((name, path, description, _signature(path, inputs, outputs)))

    # export_path was created empty, so any other file comes from the export and would be missing in the AI-BOM
    unlisted = sorted(set(os.listdir(export_path)) - {os.path.basename(path) for _, path, *_ in exported})
    if unlisted:
        raise RuntimeError(
            f"The ONNX export wrote {unlisted}, which the AI-BOM doesn't cover. "
            "ONNX external data of models larger than 2 GB isn't supported."
        )

    write_spdx_aibom(
        aibom_path=os.path.join(export_path, AIBOM_FILENAME),
        aibom_uri=aibom_uri,
        exported=exported,
        model_type=type(model).__name__,
        hyperparameters=_hyperparameters(model, config["model"].get("init_args") or {}),
        model_version=model_version,
        supplier=supplier,
        dataset=_dataset(config),
    )
