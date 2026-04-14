from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_project.stage2_preprocess.augmentation import AugmentPipeline
from ocr_project.stage2_preprocess.preprocess import PreprocessPipeline

from train.datasets.common import SampleRecord, hwc_to_chw, load_image, pad_width_to_batch, parse_rec_gt


@dataclass(slots=True)
class RecSample:
    image: np.ndarray
    label: list[int]
    label_len: int
    text: str
    text_type: str


class RecDataset:
    def __init__(
        self,
        data_dir: str | Path,
        label_file: str | Path,
        dict_path: str | Path,
        preprocess_config: str | Path,
        augment_config: str | Path,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.label_file = Path(label_file)
        self.dictionary = load_dictionary(dict_path)
        self.index_map = {char: index for index, char in enumerate(self.dictionary)}
        self.preprocess = PreprocessPipeline(preprocess_config)
        self.augment = AugmentPipeline(augment_config)
        self.records = parse_rec_gt(self.label_file)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RecSample:
        record = self.records[index]
        image = load_image(self.data_dir / Path(record.image_path).name)
        processed = self.preprocess.run(image, mode="rec")
        text_type = infer_text_type(record)
        augmented = self.augment.run_rec((processed["image"] * 255.0).astype(np.uint8), text_type=text_type)
        image_tensor = hwc_to_chw(augmented["image"] / 255.0)
        label = encode_text(record.label, self.index_map)
        return RecSample(
            image=image_tensor,
            label=label,
            label_len=len(label),
            text=record.label,
            text_type=text_type,
        )


def load_dictionary(dict_path: str | Path) -> list[str]:
    return Path(dict_path).read_text(encoding="utf-8").splitlines()


def encode_text(text: str, index_map: dict[str, int]) -> list[int]:
    unk_index = index_map.get("<unk>", 0)
    return [index_map.get(char, unk_index) for char in text]


def infer_text_type(record: SampleRecord) -> str:
    text = record.label.strip()
    if len(text) == 1:
        return "single_char"
    if re.search(r"\d", text) and any(token in text for token in ("/", ".", ":", "-")):
        return "date"
    if any(keyword in record.image_path.name.lower() for keyword in ("hand", "write")):
        return "handwrite"
    return "normal"


def rec_collate_fn(samples: list[RecSample]) -> dict[str, np.ndarray | list[int] | list[str]]:
    batch_images = pad_width_to_batch([sample.image for sample in samples], padding_value=0.0)
    max_len = max(sample.label_len for sample in samples)
    labels = np.full((len(samples), max_len), fill_value=-1, dtype=np.int64)
    label_lens: list[int] = []
    texts: list[str] = []
    text_types: list[str] = []
    for index, sample in enumerate(samples):
        labels[index, : sample.label_len] = np.array(sample.label, dtype=np.int64)
        label_lens.append(sample.label_len)
        texts.append(sample.text)
        text_types.append(sample.text_type)
    return {
        "image": batch_images,
        "label": labels,
        "label_len": np.array(label_lens, dtype=np.int64),
        "text": texts,
        "text_type": text_types,
    }
