import hashlib
import subprocess
import sys

import pytest
from huggingface_hub import ModelCard

from spherinator.models import create_model_card, upload_to_huggingface
from spherinator.models.aibom import write_spdx_aibom

REPO_ID = "HITS-AIN/spherinator-test"
DATASET_ID = "HITS-AIN/test-dataset"


def _write_export(path, dataset):
    exported = []
    for name in ("encoder", "decoder", "reconstruction"):
        onnx_file = path / f"{name}.onnx"
        onnx_file.write_bytes(name.encode())
        exported.append((name, str(onnx_file), f"{name} description"))
    write_spdx_aibom(
        aibom_path=str(path / "aibom.spdx.json"),
        aibom_uri=f"https://huggingface.co/{REPO_ID}/resolve/main/aibom.spdx.json",
        exported=exported,
        model_type="Autoencoder",
        hyperparameters={"input_shape": (2, 1, 12), "latent_shape": (2, 3), "opset_version": 19},
        model_version="1.2.3",
        supplier="HITS",
        dataset=dataset,
    )
    return path


@pytest.fixture
def export_path(tmp_path):
    return _write_export(tmp_path, (DATASET_ID, f"https://huggingface.co/datasets/{DATASET_ID}"))


@pytest.fixture
def hub_calls(monkeypatch):
    """Records the HfApi calls of upload_to_huggingface instead of contacting the Hub."""
    calls = []

    class FakeHfApi:
        def __init__(self, token=None):
            calls.append(("init", {"token": token}))

        def __getattr__(self, name):
            return lambda **kwargs: calls.append((name, kwargs))

    # Importing the submodule by name would shadow the lazily loaded function in spherinator.models
    monkeypatch.setattr(sys.modules[upload_to_huggingface.__module__], "HfApi", FakeHfApi)
    return calls


def test_create_model_card(export_path):
    card = create_model_card(str(export_path), repo_id=REPO_ID, tags=["astronomy"], description="Test model.")

    assert card.data.library_name == "onnx"
    assert card.data.license == "apache-2.0"
    assert card.data.tags == ["spherinator", "onnx", "astronomy"]
    assert card.data.model_name == "spherinator-test"
    assert card.data.datasets == [DATASET_ID]

    for name in ("encoder", "decoder", "reconstruction"):
        sha256 = hashlib.sha256((export_path / f"{name}.onnx").read_bytes()).hexdigest()
        assert f"| `{name}.onnx` | {name} description | `{sha256}` |" in card.text
    assert "Test model." in card.text
    assert "`Autoencoder`" in card.text
    assert "version 1.2.3, supplied by HITS" in card.text
    assert f'hf_hub_download("{REPO_ID}", "encoder.onnx")' in card.text
    assert "np.zeros((2, 1, 12), dtype=np.float32)" in card.text
    assert f"## Training data\n\n- [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})\n" in card.text

    assert ModelCard(str(card)).data.to_dict() == card.data.to_dict()


def test_create_model_card_local_dataset(tmp_path):
    card = create_model_card(str(_write_export(tmp_path, ("illustris", None))), repo_id=REPO_ID)

    assert card.data.datasets is None
    assert "## Training data\n\n- `illustris`\n" in card.text


def test_create_model_card_without_dataset(tmp_path):
    card = create_model_card(str(_write_export(tmp_path, None)), repo_id=REPO_ID)

    assert card.data.datasets is None
    assert "Training data" not in card.text


def test_upload_to_huggingface_with_model_card(export_path, hub_calls):
    card = create_model_card(str(export_path), repo_id=REPO_ID, path_in_repo="v1")

    assert f'hf_hub_download("{REPO_ID}", "v1/encoder.onnx")' in card.text
    assert "[`v1/aibom.spdx.json`](v1/aibom.spdx.json)" in card.text

    upload_to_huggingface(str(export_path), REPO_ID, path_in_repo="v1", token="secret", model_card=card)

    assert [name for name, _ in hub_calls] == ["init", "create_repo", "upload_folder", "upload_file"]
    assert hub_calls[0][1] == {"token": "secret"}
    assert hub_calls[2][1]["path_in_repo"] == "v1"
    upload_file = hub_calls[3][1]
    assert upload_file["path_in_repo"] == "README.md"
    assert upload_file["path_or_fileobj"] == str(card).encode("utf-8")
    assert upload_file["repo_id"] == REPO_ID
    assert upload_file["repo_type"] == "model"


def test_upload_to_huggingface_without_model_card(export_path, hub_calls):
    upload_to_huggingface(str(export_path), REPO_ID)

    assert [name for name, _ in hub_calls] == ["init", "create_repo", "upload_folder"]
    assert hub_calls[1][1]["private"] is False


def test_upload_to_huggingface_private(export_path, hub_calls):
    upload_to_huggingface(str(export_path), REPO_ID, private=True)

    assert hub_calls[1][1]["private"] is True


def test_lazy_imports_are_functions():
    # A fresh interpreter, because importing a submodule binds its name in spherinator.models
    code = (
        "import inspect\n"
        "from spherinator.models import create_model_card, export_onnx, upload_to_huggingface\n"
        "assert all(inspect.isfunction(f) for f in (create_model_card, export_onnx, upload_to_huggingface))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
