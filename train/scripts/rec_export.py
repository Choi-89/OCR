from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from ocr_project.stage6_deployment.export_model import InferenceExportSpec, export_inference_model, export_result_as_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dict_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    result = export_inference_model(
        InferenceExportSpec(
            model_key="rec",
            model_name="SVTR-Tiny",
            config_path=Path(args.config),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            input_shape=[None, 3, 32, None],
            dict_path=Path(args.dict_path),
        ),
        dry_run=args.dry_run,
    )
    pprint(export_result_as_dict(result))


if __name__ == "__main__":
    main()
