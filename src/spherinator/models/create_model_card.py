import json
import os
from urllib.parse import urlparse

from huggingface_hub import ModelCard, ModelCardData

from .aibom import AIBOM_FILENAME, HF_DATASETS_URL, SPDX_VERSION


def create_model_card(
    export_path: str,
    repo_id: str,
    license: str = "apache-2.0",
    tags: list[str] | None = None,
    description: str = "",
    path_in_repo: str = "",
) -> ModelCard:
    """Creates a Hugging Face model card for a directory written by :func:`~spherinator.models.export_onnx`.

    The files, SHA-256 hashes, version, supplier, hyperparameters, and training dataset are read from the AI-BOM
    (``aibom.spdx.json``) in ``export_path``, so the card always matches the exported ONNX files. Hugging Face Hub
    datasets are also added to the ``datasets`` metadata.

    Args:
        export_path (str): Directory containing the ONNX files and the AI-BOM.
        repo_id (str): HuggingFace repository ID, used as title and in the usage example.
        license (str, optional): License identifier of the models. Defaults to "apache-2.0".
        tags (list[str], optional): Tags added to "spherinator" and "onnx". Defaults to None.
        description (str, optional): Text shown below the title. Defaults to "".
        path_in_repo (str, optional): Repository subfolder the export is uploaded to, i.e. the ``path_in_repo``
            of :func:`~spherinator.models.upload_to_huggingface`. The card itself stays in the repository root.
            Defaults to "".

    Returns:
        ModelCard: Model card to pass to :func:`~spherinator.models.upload_to_huggingface`.
    """
    with open(os.path.join(export_path, AIBOM_FILENAME), encoding="utf-8") as f:
        graph = json.load(f)["@graph"]

    packages = [element for element in graph if element["type"] == "ai_AIPackage"]
    supplier = next(element["name"] for element in graph if element["type"] == "Organization")
    files = [os.path.basename(urlparse(package["software_downloadLocation"]).path) for package in packages]
    hashes = [
        next(h["hashValue"] for h in package["verifiedUsing"] if h["algorithm"] == "sha256") for package in packages
    ]
    hyperparameters = {entry["key"]: entry["value"] for entry in packages[0]["ai_hyperparameter"]}
    datasets = [element for element in graph if element["type"] == "dataset_DatasetPackage"]
    hub_datasets = [
        dataset["name"]
        for dataset in datasets
        if dataset.get("software_downloadLocation", "").startswith(HF_DATASETS_URL)
    ]
    model_name = repo_id.split("/")[-1]
    prefix = f"{path_in_repo.strip('/')}/" if path_in_repo.strip("/") else ""

    card_data = ModelCardData(
        license=license,
        library_name="onnx",
        tags=["spherinator", "onnx", *(tags or [])],
        model_name=model_name,
        datasets=hub_datasets or None,
    )

    intro = f"{description}\n\n" if description else ""
    dataset_rows = "\n".join(
        f"- [{dataset['name']}]({dataset['software_downloadLocation']})"
        if dataset["name"] in hub_datasets
        else f"- `{dataset['name']}`"
        for dataset in datasets
    )
    training_data = f"## Training data\n\n{dataset_rows}\n\n" if datasets else ""
    file_rows = "\n".join(
        f"| `{file}` | {package['description']} | `{sha256}` |"
        for file, package, sha256 in zip(files, packages, hashes, strict=True)
    )
    hyperparameter_rows = "\n".join(f"| `{key}` | `{value}` |" for key, value in hyperparameters.items())

    body = f"""# {model_name}

{intro}ONNX export of a [Spherinator](https://github.com/HITS-AIN/Spherinator) `{packages[0]["ai_typeOfModel"][0]}`,
version {packages[0]["software_packageVersion"]}, supplied by {supplier}.

{training_data}## Files

| File | Description | SHA-256 |
|------|-------------|---------|
{file_rows}
| `{AIBOM_FILENAME}` | SPDX {SPDX_VERSION} AI-BOM of the ONNX files | |

## Hyperparameters

| Name | Value |
|------|-------|
{hyperparameter_rows}

All graphs have a dynamic batch axis, so the first dimension of the shapes is only the batch size used for the export.

## Usage

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

session = ort.InferenceSession(hf_hub_download("{repo_id}", "{prefix}{files[0]}"))
x = np.zeros({hyperparameters["input_shape"]}, dtype=np.float32)
(output,) = session.run(None, {{"input": x}})
```

## AI-BOM

[`{prefix}{AIBOM_FILENAME}`]({prefix}{AIBOM_FILENAME}) is an [SPDX {SPDX_VERSION}](https://spdx.dev/use/specifications/) AI bill
of materials listing every ONNX file with its SHA-256 hash. Each ONNX file links to it via the `aibom.uri` and
`aibom.format` metadata properties.
"""

    return ModelCard(f"---\n{card_data.to_yaml()}\n---\n{body}")
