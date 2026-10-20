import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from urllib.parse import urljoin

AIBOM_FILENAME = "aibom.spdx.json"
SPDX_VERSION = "3.0.1"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_spdx_aibom(
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
