import pytest
from spherinator.models.aibom import write_spdx_aibom


def _write_onnx(path, name):
    onnx_file = path / f"{name}.onnx"
    onnx_file.write_bytes(name.encode())
    return (name, str(onnx_file), f"{name} description", f"{name} inputs and outputs")


@pytest.mark.parametrize(
    "exported, kwargs, match",
    [
        (lambda path: [], {}, "at least one"),
        (lambda path: [_write_onnx(path, "encoder")] * 2, {}, "names must be unique"),
        (
            lambda path: [_write_onnx(path, "encoder"), ("decoder", str(path / "encoder.onnx"), "d", "i")],
            {},
            "file names must be unique",
        ),
        (lambda path: [("encoder", str(path / "missing.onnx"), "d", "i")], {}, "doesn't exist"),
        (lambda path: [_write_onnx(path, "my encoder")], {}, "may only contain"),
        (lambda path: [(*_write_onnx(path, "encoder")[:3], " ")], {}, "must not be empty"),
        (lambda path: [_write_onnx(path, "encoder")], {"supplier": ""}, "supplier must not be empty"),
        (lambda path: [_write_onnx(path, "encoder")], {"dataset": ("", None)}, "Dataset name"),
    ],
)
def test_write_spdx_aibom_rejects_invalid_input(tmp_path, exported, kwargs, match):
    aibom_path = tmp_path / "aibom.spdx.json"
    with pytest.raises(ValueError, match=match):
        write_spdx_aibom(
            **{
                "aibom_path": str(aibom_path),
                "aibom_uri": "aibom.spdx.json",
                "exported": exported(tmp_path),
                "model_type": "Autoencoder",
                "hyperparameters": {},
                "model_version": "1.0.0",
                "supplier": "HITS",
                **kwargs,
            }
        )
    assert not aibom_path.exists()
