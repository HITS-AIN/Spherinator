import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from urllib.parse import urljoin

AIBOM_FILENAME = "aibom.spdx.json"
SPDX_VERSION = "3.0.1"
HF_DATASETS_URL = "https://huggingface.co/datasets/"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_inputs(
    exported: list[tuple[str, str, str, str]],
    model_type: str,
    model_version: str,
    supplier: str,
    dataset: tuple[str, str | None] | None,
):
    """Raises a ValueError for arguments of :func:`write_spdx_aibom` that would produce an incomplete or
    ambiguous AI-BOM."""
    if not exported:
        raise ValueError("exported must list at least one ONNX file")
    names = [name for name, *_ in exported]
    files = [os.path.basename(path) for _, path, *_ in exported]
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError(f"Name {name!r} may only contain letters, digits, '.', '_' and '-'")
    for kind, values in (("names", names), ("file names", files)):
        if len(set(values)) != len(values):
            raise ValueError(f"Exported {kind} must be unique: {values}")
    for name, path, description, information in exported:
        if not os.path.isfile(path):
            raise ValueError(f"ONNX file {path!r} of {name!r} doesn't exist")
        if not description.strip() or not information.strip():
            raise ValueError(f"Description and information about application of {name!r} must not be empty")
    for field, value in (("model_type", model_type), ("model_version", model_version), ("supplier", supplier)):
        if not value.strip():
            raise ValueError(f"{field} must not be empty")
    if dataset is not None and not dataset[0].strip():
        raise ValueError("Dataset name must not be empty")


def write_spdx_aibom(
    aibom_path: str,
    aibom_uri: str,
    exported: list[tuple[str, str, str, str]],
    model_type: str,
    hyperparameters: dict,
    model_version: str,
    supplier: str,
    dataset: tuple[str, str | None] | None = None,
):
    """Writes an SPDX 3.0.1 AI-BOM (JSON-LD) describing the exported ONNX files.

    Args:
        aibom_path (str): Output path of the AI-BOM file.
        aibom_uri (str): Link to the AI-BOM, used to resolve the download location of each model.
        exported (list): Tuples of (name, ONNX file path, description, information about application). The
            information about application describes the inputs and outputs of the graph.
        model_type (str): Type of the exported model, e.g. the class name.
        hyperparameters (dict): Key-value pairs listed as hyperparameters of each model, e.g. ``z_dim``. Omitted
            if empty.
        model_version (str): Version of the exported models.
        supplier (str): Name of the organization supplying the models.
        dataset (tuple, optional): Name and download location (or None) of the training dataset, added as
            dataset package with a ``trainedOn`` relationship from each model. Defaults to None.

    Raises:
        ValueError: If ``exported`` is empty, has duplicate or invalid names, duplicate file names or missing files,
            or if a text field is empty.
    """
    _check_inputs(exported, model_type, model_version, supplier, dataset)

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
            **(
                {
                    "ai_hyperparameter": [
                        {"type": "DictionaryEntry", "key": key, "value": str(value)}
                        for key, value in hyperparameters.items()
                    ]
                }
                if hyperparameters
                else {}
            ),
            "ai_autonomyType": "no",
            "ai_informationAboutApplication": information,
        }
        for name, path, description, information in exported
    ]
    package_ids = [package["spdxId"] for package in packages]

    dataset_elements = []
    profiles = ["core", "software", "ai"]
    if dataset is not None:
        dataset_name, dataset_location = dataset
        dataset_id = f"{namespace}#dataset"
        dataset_elements = [
            {
                "type": "dataset_DatasetPackage",
                "spdxId": dataset_id,
                "creationInfo": creation_info,
                "name": dataset_name,
                **({"software_downloadLocation": dataset_location} if dataset_location else {}),
                "software_primaryPurpose": "data",
                "dataset_datasetType": ["noAssertion"],
            },
            *(
                {
                    "type": "Relationship",
                    "spdxId": f"{package_id}-trainedOn",
                    "creationInfo": creation_info,
                    "from": package_id,
                    "relationshipType": "trainedOn",
                    "to": [dataset_id],
                }
                for package_id in package_ids
            ),
        ]
        profiles.append("dataset")

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
            *dataset_elements,
            {
                "type": "SpdxDocument",
                "spdxId": f"{namespace}#document",
                "creationInfo": creation_info,
                "name": "Spherinator ONNX AI-BOM",
                "profileConformance": profiles,
                "rootElement": package_ids,
                "element": [supplier_id, tool_id, *package_ids, *(e["spdxId"] for e in dataset_elements)],
            },
        ],
    }

    with open(aibom_path, "w", encoding="utf-8") as f:
        json.dump(aibom, f, indent=2, ensure_ascii=False)
