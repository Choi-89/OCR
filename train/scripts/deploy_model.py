from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from ocr_project.stage6_deployment.versioning import RegistryPaths, deploy_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--registry_dir", default="backend/ocr/model_registry")
    parser.add_argument("--inference_dir", default="backend/ocr/inference")
    parser.add_argument("--active_version_file", default="backend/ocr/active_version.txt")
    parser.add_argument("--registry_index", default="backend/ocr/registry_index.json")
    parser.add_argument("--deployment_log", default="backend/ocr/deployment_log.jsonl")
    parser.add_argument("--by", default="OCR team")
    args = parser.parse_args()

    result = deploy_model(
        args.version,
        RegistryPaths(
            registry_dir=Path(args.registry_dir),
            inference_dir=Path(args.inference_dir),
            active_version_file=Path(args.active_version_file),
            registry_index=Path(args.registry_index),
            deployment_log=Path(args.deployment_log),
        ),
        by=args.by,
    )
    pprint(result)


if __name__ == "__main__":
    main()
