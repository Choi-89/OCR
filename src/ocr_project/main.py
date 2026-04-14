from __future__ import annotations

from dataclasses import asdict
from pprint import pprint
from pathlib import Path

import numpy as np

from ocr_project.stage1_data.collection_spec import default_collection_spec
from ocr_project.stage1_data.annotation_guide import (
    bbox_rules,
    build_annotation_rules,
    default_annotation_policy,
    recommended_tools,
    text_input_rules,
)
from ocr_project.stage1_data.annotation_workflow import (
    OCRPolygon,
    calibration_manifest,
    evaluate_annotation_definition_of_done,
    format_det_gt_line,
    format_rec_gt_line,
    initialize_annotation_workspace,
    mean_iou,
    passes_iou_calibration,
    polygon_mode_required,
    ppocrlabel_shortcuts,
    recommended_ppocrlabel_commands,
)
from ocr_project.stage1_data.collection_workflow import (
    ImageQualitySnapshot,
    assess_image_quality,
    build_filename,
    initialize_collection_workspace,
)
from ocr_project.stage1_data.synthetic_generator import (
    SyntheticGenerationRequest,
    build_synth_filename,
    default_synth_params,
    generate_synthetic_samples,
    initialize_synthetic_workspace,
    plan_supplement_counts,
    recommend_visual_qc_sample_size,
)
from ocr_project.stage1_data.dataset_splitter import (
    build_quality_report,
    check_prerequisites,
    collect_dataset_records,
    evaluate_dataset_definition_of_done,
    initialize_dataset_workspace,
    parse_filename_metadata,
    run_automatic_quality_checks,
    stratified_split,
    summarize_dataset_stats,
    validate_split_bundle,
)
from ocr_project.stage2_preprocess.augmentation import (
    AugmentPipeline,
    AugmentationConfig,
    describe_augmentation_pipeline,
)
from ocr_project.stage2_preprocess.korean_charset import (
    build_dictionary,
    estimate_vocab_size,
    validate_dictionary,
)
from ocr_project.stage2_preprocess.preprocess import PreprocessPipeline
from ocr_project.stage3_models.detection_model import (
    DBNetPPModelSpec,
    DetectionModelConfig,
    build_detection_model,
)
from ocr_project.stage3_models.angle_classifier import (
    AngleClassifierConfig,
    MobileNetV3SmallAngleSpec,
    build_angle_classifier,
)
from ocr_project.stage3_models.recognition_model import (
    CRNNModelSpec,
    RecognitionModelConfig,
    SVTRTinyModelSpec,
    build_recognition_model,
)
from ocr_project.stage4_training.hyperparameters import default_hyperparameters
from ocr_project.stage4_training.environment import environment_summary, initialize_train_workspace
from ocr_project.stage5_evaluation.quality_gate import passes_quality_gate
from ocr_project.stage6_deployment.service_integration import integration_checklist


def main() -> None:
    """Print the current OCR project scaffold summary."""
    workspace = initialize_collection_workspace(Path.cwd())
    synth_workspace = initialize_synthetic_workspace(Path.cwd())
    annotation_workspace = initialize_annotation_workspace(Path.cwd())
    passed, reasons = assess_image_quality(
        ImageQualitySnapshot(
            width=1440,
            height=1080,
            text_area_ratio=0.55,
            tilt_degrees=8.0,
            legibility_ratio=0.97,
            file_size_bytes=850_000,
        )
    )
    supplement_plan = plan_supplement_counts(
        {
            "paper_hospital": 0,
            "screen_food": 42,
            "scan_factory": 75,
        }
    )
    generated = generate_synthetic_samples(
        SyntheticGenerationRequest(
            template_dir=Path.cwd() / "assets" / "templates",
            output_dir=synth_workspace["render_root"] / "hospital",
            label_dir=synth_workspace["label_root"],
            count=2,
            method="render",
            industry="hospital",
        )
    )
    dataset_workspace = initialize_dataset_workspace(Path.cwd())
    dataset_records = collect_dataset_records(Path.cwd())
    automatic_issues = run_automatic_quality_checks(dataset_records)
    split_bundle = stratified_split(dataset_records, seed=42)
    split_issues = validate_split_bundle(split_bundle)
    dataset_stats = summarize_dataset_stats(dataset_records)
    preprocess_pipeline = PreprocessPipeline(Path.cwd() / "configs" / "preprocess_config.yaml")
    preprocess_sample = preprocess_pipeline.run(np.full((720, 1280, 3), 255, dtype=np.uint8), "det", angle=0)
    augment_pipeline = AugmentPipeline(Path.cwd() / "configs" / "augment_config.yaml")
    augment_det_sample = augment_pipeline.run_det(
        np.full((256, 512, 3), 255, dtype=np.uint8),
        [[20, 20, 80, 60], [120, 30, 200, 90]],
        seed=42,
    )
    augment_rec_sample = augment_pipeline.run_rec(np.full((32, 128, 3), 255, dtype=np.uint8), "single_char", seed=42)
    dict_output_dir = Path.cwd() / "backend" / "ocr" / "dict"
    dict_output_dir.mkdir(parents=True, exist_ok=True)
    sample_rec_gt = Path.cwd() / "data" / "labels" / "rec_gt.txt"
    if not sample_rec_gt.exists():
        sample_rec_gt.parent.mkdir(parents=True, exist_ok=True)
        sample_rec_gt.write_text("crop/sample_0.png D\ncrop/sample_1.png 낮\n", encoding="utf-8")
    dict_result = build_dictionary(sample_rec_gt, dict_output_dir)
    dict_validation = validate_dictionary(dict_result.paths.versioned_dict, sample_rec_gt)
    train_workspace = initialize_train_workspace(Path.cwd())
    det_spec = DBNetPPModelSpec(DetectionModelConfig())
    cls_spec = MobileNetV3SmallAngleSpec(AngleClassifierConfig())
    rec_spec = SVTRTinyModelSpec(RecognitionModelConfig())
    rec_crnn_spec = CRNNModelSpec()
    sample_polygons = [
        OCRPolygon(transcription="D", points=[[12, 34], [56, 34], [56, 60], [12, 60]]),
        OCRPolygon(transcription="09:00", points=[[60, 34], [120, 34], [120, 60], [60, 60]]),
    ]
    summary = {
        "collection_spec": default_collection_spec(),
        "sample_filename": build_filename("paper", "hospital", 1, ".jpg"),
        "workspace": workspace,
        "synthetic_workspace": synth_workspace,
        "annotation_workspace": annotation_workspace,
        "dataset_workspace": dataset_workspace,
        "sample_synth_filename": build_synth_filename("render", "hospital", 1),
        "supplement_plan": supplement_plan,
        "synthetic_defaults": default_synth_params(Path.cwd()),
        "visual_qc_sample_size": recommend_visual_qc_sample_size(400),
        "generated_synthetic_files": [str(path) for path in generated],
        "dataset_prerequisites": check_prerequisites(Path.cwd()),
        "parsed_filename_example": parse_filename_metadata("paper_hospital_0001.jpg"),
        "dataset_record_count": len(dataset_records),
        "automatic_quality_issues": [asdict(issue) for issue in automatic_issues[:10]],
        "dataset_stats": dataset_stats,
        "split_counts": {
            "train": len(split_bundle.train),
            "val": len(split_bundle.val),
            "test": len(split_bundle.test),
        },
        "split_issues": split_issues,
        "quality_report_preview": build_quality_report(
            automatic_issues=automatic_issues[:5],
            manual_review_error_rate=0.02,
            stats={"all": dataset_stats},
            split_issues=split_issues,
        ),
        "dataset_dod_example": evaluate_dataset_definition_of_done(
            automatic_issue_count=len(automatic_issues),
            manual_review_error_rate=0.02,
            split_issues=split_issues,
            split_config_exists=False,
            dataset_stats_exists=False,
            quality_report_exists=False,
            frozen=False,
        ),
        "preprocess_config_path": str(Path.cwd() / "configs" / "preprocess_config.yaml"),
        "preprocess_sample": {
            "image_shape": preprocess_sample["image"].shape,
            "original_shape": preprocess_sample["original_shape"],
            "padded_shape": preprocess_sample["padded_shape"],
            "padding": preprocess_sample["padding"],
            "angle_corrected": preprocess_sample["angle_corrected"],
            "deskew_angle": preprocess_sample["deskew_angle"],
            "processing_ms": preprocess_sample["processing_ms"],
        },
        "augment_config_path": str(Path.cwd() / "configs" / "augment_config.yaml"),
        "augment_det_sample": {
            "image_shape": augment_det_sample["image"].shape,
            "bbox_count": len(augment_det_sample["bboxes"]),
            "applied": augment_det_sample["applied"],
        },
        "augment_rec_sample": {
            "image_shape": augment_rec_sample["image"].shape,
            "applied": augment_rec_sample["applied"],
        },
        "dictionary_output_dir": str(dict_output_dir),
        "dictionary_vocab_size": estimate_vocab_size(),
        "dictionary_files": {
            "versioned_dict": str(dict_result.paths.versioned_dict),
            "latest_dict": str(dict_result.paths.latest_dict),
            "meta_json": str(dict_result.paths.meta_json),
            "freq_txt": str(dict_result.paths.freq_txt),
            "chars_from_annotation": str(dict_result.paths.chars_from_annotation),
        },
        "dictionary_validation": dict_validation,
        "train_workspace": train_workspace,
        "training_environment": environment_summary(),
        "det_model_config_path": str(Path.cwd() / "backend" / "ocr" / "models" / "det" / "det_config.yaml"),
        "det_model_summary": det_spec.model_summary((1, 3, 960, 960)),
        "cls_model_config_path": str(Path.cwd() / "backend" / "ocr" / "models" / "cls" / "cls_config.yaml"),
        "cls_model_summary": cls_spec.model_summary((1, 3, 48, 192)),
        "rec_model_config_path": str(Path.cwd() / "backend" / "ocr" / "models" / "rec" / "rec_config.yaml"),
        "rec_model_summary": rec_spec.model_summary((4, 3, 32, 128)),
        "rec_crnn_summary": rec_crnn_spec.model_summary((4, 3, 32, 128)),
        "annotation_tools": recommended_tools(),
        "ppocrlabel_commands": recommended_ppocrlabel_commands(),
        "ppocrlabel_shortcuts": ppocrlabel_shortcuts(),
        "annotation_policy": default_annotation_policy(),
        "bbox_rules": bbox_rules(),
        "text_input_rules": text_input_rules(),
        "det_gt_example": format_det_gt_line("img/paper_hospital_0001.jpg", sample_polygons),
        "rec_gt_example": format_rec_gt_line("crop/paper_hospital_0001_0.jpg", "D"),
        "polygon_required_example": polygon_mode_required(18.0),
        "calibration_manifest": calibration_manifest(),
        "mean_iou_example": mean_iou([([0, 0, 10, 10], [0, 0, 10, 10])]),
        "passes_iou_calibration": passes_iou_calibration([([0, 0, 10, 10], [0, 0, 10, 10])]),
        "annotation_dod_example": evaluate_annotation_definition_of_done(
            masked_image_count=100,
            det_gt_exists=True,
            rec_gt_exists=True,
            crop_count=300,
            unreadable_box_count=5,
            total_box_count=200,
            mean_calibration_iou=0.9,
            cross_reviewed_image_count=10,
            annotation_log_exists=True,
        ),
        "quality_assessment": {"passed": passed, "reasons": reasons},
        "annotation_rules": build_annotation_rules(),
        "augmentation": describe_augmentation_pipeline(AugmentationConfig()),
        "detection_model": build_detection_model(DetectionModelConfig()),
        "angle_classifier": build_angle_classifier(AngleClassifierConfig()),
        "recognition_model": build_recognition_model(RecognitionModelConfig()),
        "hyperparameters": default_hyperparameters(),
        "quality_gate_example": passes_quality_gate(cer=0.025),
        "service_integration": integration_checklist(),
    }
    pprint(summary)


if __name__ == "__main__":
    main()
