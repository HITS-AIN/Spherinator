import hashlib
import json
import sys

import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
import yaml
from power_spherical import PowerSpherical
from spdx3_validate import validate
from spherinator.models import (
    Autoencoder,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
    VariationalAutoencoder,
    export_onnx,
)
from torch import nn

# Importing the submodule by name would shadow the lazily loaded function in spherinator.models
_signature = sys.modules[export_onnx.__module__]._signature


def assert_valid_spdx(path):
    """Validates an SPDX 3 document against the SPDX JSON Schema and SHACL model."""
    result = validate(str(path))
    assert result, str(result)


class Model1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DistributionModel(torch.nn.Module):
    def __init__(self, dist):
        self.dist = dist
        super().__init__()

    def forward(self, x):
        return self.dist.rsample(x.shape)


@pytest.mark.parametrize(
    ("module", "input"),
    [
        (Model1(), torch.randn(2, 2, 2)),
        (Model2(), torch.randn(1, 1, 32, 32)),
        pytest.param(
            DistributionModel(PowerSpherical(torch.Tensor([0.0, 1.0]), torch.Tensor([1.0]))),
            torch.randn(2, 3),
            marks=pytest.mark.xfail,
        ),
        (
            ConvolutionalEncoder1D([1, 12], 3),
            torch.randn(2, 1, 12),
        ),
        pytest.param(
            Autoencoder(
                ConvolutionalEncoder1D([1, 12], 3),
                ConvolutionalDecoder1D(3, [1, 12], [1, 12]),
            ),
            torch.randn(2, 1, 12),
            marks=pytest.mark.xfail(sys.version_info.minor == 9, reason="Fails on Python 3.9"),
        ),
    ],
)
def test_onnx_dynamo_export(module, input):
    module.eval()
    module(input)
    torch.onnx.export(module, (input,), dynamo=True)


def test_package_exporter_produces_dynamic_equivalent_models(tmp_path):
    config = {
        "model": {
            "class_path": "spherinator.models.Autoencoder",
            "init_args": {
                "encoder": {
                    "class_path": "spherinator.models.ConvolutionalEncoder1D",
                    "init_args": {"input_dim": [1, 12], "output_dim": 3},
                },
                "decoder": {
                    "class_path": "spherinator.models.ConvolutionalDecoder1D",
                    "init_args": {"input_dim": 3, "output_dim": [1, 12], "cnn_input_dim": [1, 12]},
                },
            },
        }
    }
    model = Autoencoder(ConvolutionalEncoder1D([1, 12], 3), ConvolutionalDecoder1D(3, [1, 12], [1, 12]))
    model.eval()
    model(torch.randn(2, 1, 12))
    checkpoint = tmp_path / "model.ckpt"
    config_path = tmp_path / "model.yaml"
    export_path = tmp_path / "onnx"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    export_onnx(str(checkpoint), str(config_path), str(export_path), (2, 1, 12), (2, 3))

    for name in ("encoder", "decoder", "reconstruction"):
        onnx.checker.check_model(onnx.load(export_path / f"{name}.onnx"))
        session = ort.InferenceSession(export_path / f"{name}.onnx", providers=["CPUExecutionProvider"])
        assert len(session.get_outputs()) == 1

    assert_valid_spdx(export_path / "aibom.spdx.json")
    aibom = json.loads((export_path / "aibom.spdx.json").read_text())
    assert not any(e["type"] in ("dataset_DatasetPackage", "Relationship") for e in aibom["@graph"])
    packages = {p["name"]: p for p in aibom["@graph"] if p["type"] == "ai_AIPackage"}
    for package in packages.values():
        assert "concentration" not in package["description"]
        assert "concentration" not in package["ai_informationAboutApplication"]
    assert packages["Spherinator encoder"]["description"] == "Backbone → latent vector"
    assert packages["Spherinator encoder"]["ai_informationAboutApplication"] == (
        "Inputs: input float32[batch, 1, 12] (input data). Outputs: embedding float32[batch, 3] (latent vector)."
    )

    session = ort.InferenceSession(export_path / "reconstruction.onnx", providers=["CPUExecutionProvider"])
    for batch_size in (1, 3):
        inputs = torch.randn(batch_size, 1, 12)
        actual = session.run(None, {"input": inputs.numpy()})[0]
        expected = model.reconstruct(inputs).detach().numpy()
        assert actual.shape == expected.shape
        assert torch.allclose(torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-5, rtol=1e-4)


def test_export_onnx_variational(tmp_path):
    """Encoder and reconstruction graphs of a VAE also output the concentration."""
    config = {
        "model": {
            "class_path": "spherinator.models.VariationalAutoencoder",
            "init_args": {
                "encoder": {
                    "class_path": "spherinator.models.ConvolutionalEncoder1D",
                    "init_args": {"input_dim": [1, 12], "output_dim": 3},
                },
                "decoder": {
                    "class_path": "spherinator.models.ConvolutionalDecoder1D",
                    "init_args": {"input_dim": 3, "output_dim": [1, 12], "cnn_input_dim": [1, 12]},
                },
                "encoder_out_dim": 3,
                "z_dim": 3,
            },
        }
    }
    model = VariationalAutoencoder(
        ConvolutionalEncoder1D([1, 12], 3), ConvolutionalDecoder1D(3, [1, 12], [1, 12]), encoder_out_dim=3
    )
    model(torch.randn(2, 1, 12))
    model.eval()
    checkpoint = tmp_path / "model.ckpt"
    config_path = tmp_path / "model.yaml"
    export_path = tmp_path / "onnx"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    export_onnx(str(checkpoint), str(config_path), str(export_path), (2, 1, 12), (2, 3))

    expected_outputs = {
        "encoder": ["embedding", "concentration"],
        "decoder": ["reconstruction"],
        "reconstruction": ["reconstruction", "concentration"],
    }
    sessions = {}
    for name, output_names in expected_outputs.items():
        onnx.checker.check_model(onnx.load(export_path / f"{name}.onnx"))
        sessions[name] = ort.InferenceSession(export_path / f"{name}.onnx", providers=["CPUExecutionProvider"])
        assert [output.name for output in sessions[name].get_outputs()] == output_names

    assert_valid_spdx(export_path / "aibom.spdx.json")
    aibom = json.loads((export_path / "aibom.spdx.json").read_text())
    packages = {p["name"].removeprefix("Spherinator "): p for p in aibom["@graph"] if p["type"] == "ai_AIPackage"}
    assert packages["encoder"]["description"].endswith("and concentration")
    assert packages["reconstruction"]["description"].endswith("and concentration")
    assert "concentration" not in packages["decoder"]["description"]

    info = {name: package["ai_informationAboutApplication"] for name, package in packages.items()}
    assert info["encoder"].startswith("Inputs: input float32[batch, 1, 12] (input data). Outputs: ")
    assert "embedding float32[batch, 3] (unit-normalised location vector on the hypersphere)" in info["encoder"]
    assert "concentration float32[batch] (scale κ ≥ 1 of the Power Spherical distribution" in info["encoder"]
    assert info["decoder"] == (
        "Inputs: input float32[batch, 3] (latent vector). "
        "Outputs: reconstruction float32[batch, 1, 12] (reconstructed input)."
    )
    outputs = "Outputs: reconstruction float32[batch, 1, 12] (reconstructed input), concentration float32[batch]"
    assert outputs in info["reconstruction"]

    def assert_close(actual, expected):
        assert actual.shape == expected.shape
        assert torch.allclose(torch.from_numpy(actual), expected.detach(), atol=1e-5, rtol=1e-4)

    for batch_size in (1, 3):
        inputs = torch.randn(batch_size, 1, 12)
        z_location, z_scale = model.encode(inputs)
        concentration = z_scale.squeeze(-1)

        embedding, encoder_concentration = sessions["encoder"].run(None, {"input": inputs.numpy()})
        assert_close(embedding, z_location)
        assert_close(encoder_concentration, concentration)
        assert encoder_concentration.shape == (batch_size,)
        assert (encoder_concentration >= 1).all()

        reconstruction, reconstruction_concentration = sessions["reconstruction"].run(None, {"input": inputs.numpy()})
        assert_close(reconstruction, model.reconstruct(inputs))
        assert_close(reconstruction_concentration, concentration)


def _write_autoencoder(tmp_path):
    """Writes checkpoint and config of an untrained Autoencoder and returns their paths."""
    config = {
        "model": {
            "class_path": "spherinator.models.Autoencoder",
            "init_args": {
                "encoder": {
                    "class_path": "spherinator.models.ConvolutionalEncoder1D",
                    "init_args": {"input_dim": [1, 12], "output_dim": 3},
                },
                "decoder": {
                    "class_path": "spherinator.models.ConvolutionalDecoder1D",
                    "init_args": {"input_dim": 3, "output_dim": [1, 12], "cnn_input_dim": [1, 12]},
                },
            },
        }
    }
    model = Autoencoder(ConvolutionalEncoder1D([1, 12], 3), ConvolutionalDecoder1D(3, [1, 12], [1, 12]))
    model(torch.randn(2, 1, 12))
    checkpoint = tmp_path / "model.ckpt"
    config_path = tmp_path / "model.yaml"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(checkpoint), str(config_path)


def test_export_onnx_requires_is_variational(tmp_path, monkeypatch):
    """Models that don't tell whether they are variational are rejected before anything is written."""
    checkpoint, config_path = _write_autoencoder(tmp_path)
    monkeypatch.delattr(Autoencoder, "is_variational")
    export_path = tmp_path / "onnx"

    with pytest.raises(TypeError, match="is_variational"):
        export_onnx(checkpoint, config_path, str(export_path), (2, 1, 12), (2, 3))
    assert not export_path.exists()


def test_export_onnx_rejects_external_data(tmp_path, monkeypatch):
    """ONNX external data files aren't listed in the AI-BOM, so the export fails instead of writing it."""
    checkpoint, config_path = _write_autoencoder(tmp_path)
    save = torch.onnx.ONNXProgram.save
    monkeypatch.setattr(
        torch.onnx.ONNXProgram, "save", lambda self, destination, **kwargs: save(self, destination, external_data=True)
    )
    export_path = tmp_path / "onnx"

    with pytest.raises(RuntimeError, match="external data"):
        export_onnx(checkpoint, config_path, str(export_path), (2, 1, 12), (2, 3))
    assert not (export_path / "aibom.spdx.json").exists()


def test_signature_rejects_unexpected_outputs(tmp_path):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["y"])],
        "identity",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", 3])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["batch", 3])],
    )
    path = tmp_path / "identity.onnx"
    onnx.save(onnx.helper.make_model(graph), path)

    assert _signature(str(path), {"input": "input data"}, {"y": "output"}) == (
        "Inputs: input float32[batch, 3] (input data). Outputs: y float32[batch, 3] (output)."
    )
    with pytest.raises(RuntimeError, match=r"has outputs \['y'\], expected \['embedding', 'concentration'\]"):
        _signature(str(path), {"input": "input data"}, {"embedding": "", "concentration": ""})


@pytest.mark.parametrize(
    "path, dataset_name, location",
    [
        ("HITS-AIN/test-dataset", "HITS-AIN/test-dataset", "https://huggingface.co/datasets/HITS-AIN/test-dataset"),
        # Missing local paths that look like owner/name must not be taken for Hub datasets
        ("data/train.parquet", "train.parquet", None),
        ("../training-data", "training-data", None),
    ],
)
def test_export_onnx_aibom(tmp_path, monkeypatch, path, dataset_name, location):
    monkeypatch.chdir(tmp_path)
    model = Autoencoder(ConvolutionalEncoder1D([1, 12], 3), ConvolutionalDecoder1D(3, [1, 12], [1, 12]))
    model(torch.randn(2, 1, 12))
    ckpt_file = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_file)

    model_file = tmp_path / "model.yaml"
    model_file.write_text(
        f"""
model:
  class_path: spherinator.models.Autoencoder
  init_args:
    encoder:
      class_path: spherinator.models.ConvolutionalEncoder1D
      init_args:
        input_dim: [1, 12]
        output_dim: 3
    decoder:
      class_path: spherinator.models.ConvolutionalDecoder1D
      init_args:
        input_dim: 3
        output_dim: [1, 12]
        cnn_input_dim: [1, 12]
data:
  class_path: spherinator.data.DataModule
  init_args:
    path: {path}
"""
    )

    export_path = tmp_path / "export"
    export_onnx(str(ckpt_file), str(model_file), str(export_path), (2, 1, 12), (2, 3))

    assert_valid_spdx(export_path / "aibom.spdx.json")
    aibom = json.loads((export_path / "aibom.spdx.json").read_text())
    packages = {p["name"]: p for p in aibom["@graph"] if p["type"] == "ai_AIPackage"}
    assert len(packages) == 3

    (dataset,) = (e for e in aibom["@graph"] if e["type"] == "dataset_DatasetPackage")
    assert dataset["name"] == dataset_name
    assert dataset.get("software_downloadLocation") == location
    trained_on = {e["from"]: e["to"] for e in aibom["@graph"] if e["type"] == "Relationship"}
    assert trained_on == {p["spdxId"]: [dataset["spdxId"]] for p in packages.values()}

    (document,) = (e for e in aibom["@graph"] if e["type"] == "SpdxDocument")
    assert "dataset" in document["profileConformance"]
    assert dataset["spdxId"] in document["element"]

    for name in ["encoder", "decoder", "reconstruction"]:
        onnx_file = export_path / f"{name}.onnx"
        package = packages[f"Spherinator {name}"]
        assert package["verifiedUsing"][0]["hashValue"] == hashlib.sha256(onnx_file.read_bytes()).hexdigest()
        assert package["software_downloadLocation"] == f"{name}.onnx"

        props = {p.key: p.value for p in onnx.load(onnx_file, load_external_data=False).metadata_props}
        assert props["aibom.uri"] == "aibom.spdx.json"


def test_export_onnx_aibom_local_dataset(tmp_path):
    """Configs of models trained with the removed ParquetDataModule name the dataset in ``data_directory``."""
    model = Autoencoder(ConvolutionalEncoder1D([1, 12], 3), ConvolutionalDecoder1D(3, [1, 12], [1, 12]))
    model(torch.randn(2, 1, 12))
    ckpt_file = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_file)

    model_file = tmp_path / "model.yaml"
    model_file.write_text(
        f"""
model:
  class_path: spherinator.models.Autoencoder
  init_args:
    encoder:
      class_path: spherinator.models.ConvolutionalEncoder1D
      init_args:
        input_dim: [1, 12]
        output_dim: 3
    decoder:
      class_path: spherinator.models.ConvolutionalDecoder1D
      init_args:
        input_dim: 3
        output_dim: [1, 12]
        cnn_input_dim: [1, 12]
data:
  class_path: spherinator.data.ParquetDataModule
  init_args:
    data_directory: {tmp_path / "parquet"}/
"""
    )

    export_path = tmp_path / "export"
    export_onnx(str(ckpt_file), str(model_file), str(export_path), (2, 1, 12), (2, 3))

    assert_valid_spdx(export_path / "aibom.spdx.json")
    aibom = json.loads((export_path / "aibom.spdx.json").read_text())
    (dataset,) = (e for e in aibom["@graph"] if e["type"] == "dataset_DatasetPackage")
    assert dataset["name"] == "parquet"
    assert "software_downloadLocation" not in dataset
