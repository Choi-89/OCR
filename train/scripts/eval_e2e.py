from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint

from ocr_project.stage5_evaluation.e2e_metrics import E2EEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ocr_results", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    evaluator = E2EEvaluator(args.gt_dir)
    ocr_results = load_ocr_results(args.ocr_results)
    results = evaluator.evaluate_dataset(args.image_dir, ocr_results)
    outputs = evaluator.generate_report(results, args.output_dir)
    pprint({"summary": results["summary"], "outputs": outputs})


def load_ocr_results(path: str | Path) -> dict[str, list[dict]]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    output: dict[str, list[dict]] = {}
    for item in payload:
        output[item["image"]] = item.get("tokens", [])
    return output


if __name__ == "__main__":
    main()
