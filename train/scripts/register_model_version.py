from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from ocr_project.stage6_deployment.versioning import RegistryPaths, create_version_json, register_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--source_inference_dir", default="backend/ocr/inference")
    parser.add_argument("--config_dir", default="train/configs")
    parser.add_argument("--registry_dir", default="backend/ocr/model_registry")
    parser.add_argument("--registry_index", default="backend/ocr/registry_index.json")
    parser.add_argument("--released_by", default="OCR team")
    parser.add_argument("--changes", default="")
    args = parser.parse_args()

    target = register_version(
        args.version,
        args.source_inference_dir,
        args.config_dir,
        RegistryPaths(registry_dir=Path(args.registry_dir), registry_index=Path(args.registry_index)),
        version_info=create_version_json(args.version, released_by=args.released_by, changes=args.changes),
    )
    pprint({"registered": str(target)})


if __name__ == "__main__":
    main()
