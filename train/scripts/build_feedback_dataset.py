from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from ocr_project.stage6_deployment.confidence_ui import build_feedback_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_log", required=True)
    parser.add_argument("--crops_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_feedback_count", type=int, default=50)
    args = parser.parse_args()

    stats = build_feedback_dataset(
        feedback_log=Path(args.feedback_log),
        crops_dir=Path(args.crops_dir),
        output_dir=Path(args.output_dir),
        min_feedback_count=args.min_feedback_count,
    )
    pprint(stats)


if __name__ == "__main__":
    main()
