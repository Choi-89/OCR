from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.ocr.models.rec.rec_model import dummy_forward_check as svtr_forward_check, freeze_strategy
from backend.ocr.models.rec.rec_model_crnn import dummy_forward_check as crnn_forward_check
from backend.ocr.models.rec.rec_postprocess import CTCDecoderConfig, decode_logits
from ocr_project.stage3_models.recognition_model import CRNNModelSpec, RecognitionModelConfig, SVTRTinyModelSpec


CONFIG_PATH = Path("C:/OCR/backend/ocr/models/rec/rec_config.yaml")


def test_svtr_forward_shape_matches_expected_time_steps() -> None:
    spec = SVTRTinyModelSpec(RecognitionModelConfig())
    outputs = spec.forward_shape((4, 3, 32, 128))
    assert outputs["logits"] == (4, 32, spec.config.head.vocab_size)


def test_crnn_forward_shape_matches_expected_time_steps() -> None:
    spec = CRNNModelSpec()
    outputs = spec.forward_shape((4, 3, 32, 128))
    assert outputs["logits"][0] == 4
    assert outputs["logits"][1] == 32


def test_svtr_dummy_forward_returns_weight_path() -> None:
    artifacts = svtr_forward_check(CONFIG_PATH)
    assert artifacts.output_shapes["logits"][1] == 32
    assert "svtr_tiny_pretrained.pdparams" in str(artifacts.weight_path)


def test_crnn_dummy_forward_runs() -> None:
    artifacts = crnn_forward_check((4, 3, 32, 128))
    assert artifacts.output_shapes["logits"][1] == 32


def test_freeze_strategy_has_two_phases() -> None:
    strategy = freeze_strategy(CONFIG_PATH)
    assert len(strategy) == 2
    assert strategy[0]["patch_embedding_frozen"] is True
    assert strategy[1]["patch_embedding_frozen"] is False


def test_ctc_decode_supports_single_char_truncation() -> None:
    dictionary = ["A", "B", "<unk>", "<pad>", "<blank>"]
    logits = np.array(
        [
            [
                [5.0, 1.0, 0.0, 0.0, -1.0],
                [5.0, 1.0, 0.0, 0.0, -1.0],
                [0.0, 5.0, 0.0, 0.0, -1.0],
            ]
        ],
        dtype=np.float32,
    )
    result = decode_logits(
        logits,
        dictionary,
        CTCDecoderConfig(method="greedy", blank_index=4),
        text_type="single_char",
        max_output_chars=1,
    )
    assert result["texts"][0] == "A"
