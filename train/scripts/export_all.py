from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from ocr_project.stage6_deployment.export_model import export_all_models, export_result_as_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    results = export_all_models(Path(args.root), dry_run=args.dry_run)
    pprint({key: export_result_as_dict(result) for key, result in results.items()})


if __name__ == "__main__":
    main()
