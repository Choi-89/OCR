from __future__ import annotations

import argparse
from pprint import pprint

from ocr_project.stage5_evaluation.quality_gate import check_quality_gate, generate_quality_outputs, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_result", required=True)
    parser.add_argument("--rec_result", required=True)
    parser.add_argument("--cls_result", required=True)
    parser.add_argument("--e2e_result", required=True)
    parser.add_argument("--service_result", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--approved_by", default="OCR team")
    parser.add_argument("--generate_report", action="store_true")
    args = parser.parse_args()

    gate_result = check_quality_gate(
        det_summary=load_json(args.det_result),
        rec_summary=load_json(args.rec_result),
        cls_summary=load_json(args.cls_result),
        e2e_summary=load_json(args.e2e_result),
        service_summary=load_json(args.service_result) if args.service_result else {},
    )
    outputs = generate_quality_outputs(gate_result, args.output_dir, version=args.version, approved_by=args.approved_by)
    pprint({"final_status": gate_result["final_status"], "outputs": outputs})


if __name__ == "__main__":
    main()
