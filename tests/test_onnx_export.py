import sys

import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
import yaml
from power_spherical import PowerSpherical
from torch import nn

from spherinator.models import Autoencoder, ConvolutionalDecoder1D, ConvolutionalEncoder1D, export_onnx


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

    session = ort.InferenceSession(export_path / "reconstruction.onnx", providers=["CPUExecutionProvider"])
    for batch_size in (1, 3):
        inputs = torch.randn(batch_size, 1, 12)
        actual = session.run(None, {"input": inputs.numpy()})[0]
        expected = model.reconstruct(inputs).detach().numpy()
        assert actual.shape == expected.shape
        assert torch.allclose(torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-5, rtol=1e-4)
