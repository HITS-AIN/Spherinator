import pytest
import torch
from power_spherical import PowerSpherical


@pytest.mark.xfail(
    reason="RuntimeError: Encountered autograd state manager op",
)
def test_dynamo_export_normal(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            self.normal = torch.distributions.normal.Normal(0, 1)
            super().__init__()

        def forward(self, x):
            return self.normal.sample(x.shape)

    x = torch.randn(2, 3)
    exported_program = torch.export.export(Model(), args=(x,))
    onnx_program = torch.onnx.dynamo_export(
        exported_program,
        x,
    )
    onnx_program.save(str(tmp_path / "normal.onnx"))


@pytest.mark.xfail(reason="not supported feature of ONNX")
def test_dynamo_export_spherical():
    class Model(torch.nn.Module):
        def __init__(self):
            self.spherical = PowerSpherical(
                torch.Tensor([0.0, 1.0]), torch.Tensor([1.0])
            )
            super().__init__()

        def forward(self, x):
            return self.spherical.sample(x.shape)

    x = torch.randn(2, 3)
    exported_program = torch.export.export(Model(), args=(x,))
    _ = torch.onnx.dynamo_export(
        exported_program,
        x,
    )
