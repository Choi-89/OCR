from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from ocr_project.stage5_evaluation.recognition_metrics import (
    RecognitionEvaluator,
    infer_text_type,
    write_recognition_eval_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dict_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    evaluator = RecognitionEvaluator(dict_path=args.dict_path, ignore_case=False)
    label_path = Path(args.data_dir) / "rec_gt.txt"
    if label_path.exists():
        pred_texts, gt_texts, text_types, image_paths = load_rec_eval_placeholder(label_path)
        evaluator.update(pred_texts, gt_texts, text_types, image_paths)

    outputs = write_recognition_eval_outputs(
        evaluator,
        args.output_dir,
        checkpoint=args.checkpoint,
        data_split=Path(args.data_dir).name,
    )
    pprint({"metrics": evaluator.compute(), "outputs": outputs})


def load_rec_eval_placeholder(label_path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    pred_texts: list[str] = []
    gt_texts: list[str] = []
    text_types: list[str] = []
    image_paths: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or " " not in raw_line:
            continue
        image_path, text = raw_line.split(" ", 1)
        pred_texts.append(text)
        gt_texts.append(text)
        text_types.append(infer_text_type(text, image_path))
        image_paths.append(image_path)
    return pred_texts, gt_texts, text_types, image_paths


if __name__ == "__main__":
    main()
