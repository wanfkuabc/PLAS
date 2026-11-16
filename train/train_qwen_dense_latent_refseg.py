"""llava6-3-8 Stage-2 training with text-spatial attention and smooth ranking.

Phase A (``latent_warmup``) never runs SAM and uses latent MSE + relative
latent error + head BCE + head Dice.  Phase B uses latent MSE + relative
latent error + final-mask BCE + final-mask Dice + smooth same-category ranking.

Changes from llava6-3-3, each one driven by a measurement:

1. The no-target head is gone.  RefCOCO/+/g contain no no-target expressions,
   and freezing that head while training all of its inputs destroyed it: its
   output collapsed from "93.6% of positives below p=0.05" to "100% of
   positives inside [0.30, 0.70)", i.e. no discriminative power at all.
2. The image-free ``prompt_calibrator`` is gone.  Its gate trained *down*
   (0.050 -> 0.038) while the image-guided one tripled (0.050 -> 0.181): the
   model itself voted it useless, and it cost ~1.6M parameters per step.
3. ``prompt_adapter_learning_rate`` 1e-5 -> 5e-5.  At 1e-5 the adapter's
   delta_scale did not move at all (0.1875 -> 0.1875 to four decimals), so
   "unfreeze the Prompt Adapter" was a no-op.
4. ``qwen_image_size`` defaults to 768: 24x24 visual tokens instead of 16x16,
   which DetailFusion resamples onto the 32x32 latent grid.

Training writes ``training_summary.json`` next to the checkpoint: the loss
terms, the quantities derived from them, and the learned gates, so the next
iteration starts from measurements instead of guesses.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

import torch
import transformers
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, Sampler

from PLAS.model.mask_latent_teacher import MaskLatentTeacher
from PLAS.model.qwen_dense_latent_refseg import QWEN_LATENT_GRID_TOKENS, QwenLatentStudent
from PLAS.train.qwen_refseg_data import GROUP_EMPTY, GROUP_SINGLE, OriginalMaskReferDataset
from PLAS.train.qwen_sam_geometry import sam_aligned_canvas, sam_image_input, sam_resize_size


@dataclass
class ModelArguments:
    stage: str = "student"
    training_phase: str = "task_adaptation"
    qwen_model_path: str = "./PLAS/pretrained/Qwen3-VL-8B-Instruct"
    sam_path: str = "./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    teacher_checkpoint: str = ""
    student_checkpoint: str = ""
    qwen_lora_r: int = 64
    qwen_lora_alpha: int = 128
    qwen_lora_dropout: float = 0.05
    qwen_lora_target_mlp: bool = True
    readout_layers: str = "-1,-9,-17"
    latent_width: int = 512
    latent_depth: int = 4
    text_cross_attention_heads: int = 8
    latent_weight: float = 1.0
    relative_latent_weight: float = 0.1
    head_bce_weight: float = 1.0
    head_dice_weight: float = 1.0
    mask_bce_weight: float = 2.0
    mask_dice_weight: float = 2.0
    smooth_rank_weight: float = 0.5
    smooth_rank_margin: float = 0.20
    smooth_rank_temperature: float = 0.05
    empty_loss_weight: float = 0.5
    mask_threshold: float = 0.55
    qwen_lora_learning_rate: float = 2e-5
    adapter_learning_rate: float = 1e-4
    calibrator_scale_learning_rate: float = 1e-3
    prompt_adapter_learning_rate: float = 5e-5
    # Keep the original recipe's effective scale LR unless a new script opts
    # into the dedicated, larger gate learning rate.
    prompt_adapter_scale_learning_rate: float = 5e-5
    decoder_half_learning_rate: float = 1e-5
    prompt_calibrator_max_scale: float = 0.50
    image_prompt_calibrator_width: int = 128
    image_prompt_calibrator_depth: int = 2
    image_prompt_calibrator_initial_scale: float = 0.05
    phase_a_sam_mask_weight: float = 0.0
    phase_a_sam_bce_weight: float = 0.5
    phase_a_sam_dice_weight: float = 0.5
    decoder_only: bool = False
    freeze_sam_decoder: bool = False
    sam_decoder_tuning: str = "student_b"
    fp32_prompt_adapter: bool = False
    dedicated_prompt_scale_optimizer: bool = False
    qwen_image_size: int = 768
    use_flash_attention_2: bool = True
    fp32_frozen_latent_modules: bool = True


@dataclass
class DataArguments:
    data_root: str = "./PLAS/datasets/refer_seg"
    refer_dataset: str = "grefcoco"
    refer_split_by: str = "unc"
    train_datasets: str = ""
    train_split_bys: str = ""
    train_split: str = "train"
    train_subset_ratio: float = 1.0
    train_subset_seed: int = 3407
    # RefCOCO/+/g carry no no-target expressions; kept only so a mixed-source
    # dataset with empties still trains sanely.
    positive_per_empty: int = 3
    # Mixed training samples each source with equal probability.  Set to
    # ``proportional`` to recover the concatenated-dataset behavior.
    dataset_sampling: str = "balanced"
    # Phase A sets this to False automatically: it never decodes through SAM.
    load_sam_images: bool = True
    # Verified negatives synthesised from COCO's exhaustive category labels.
    # 0.15 roughly doubles the no-target supply without repeating expressions.
    synthetic_negative_ratio: float = 0.0
    hard_negative_cache: str = ""
    hard_negative_max: int = 4
    reasonseg_root: str = "./PLAS/datasets/ReasonSeg"
    reasonseg_text_index: int = 0
    reasonseg_prompt_mode: str = "question"
    reasonseg_expected_count: int = -1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    dense_mask_log_details: bool = field(default=False)
    dense_mask_abort_loss: float = field(default=100.0)


def qwen_prompt(expression: str) -> str:
    return (
        "Identify the exact image region referred to by the expression below. "
        "Use only visible image evidence.\n"
        f"Expression: {expression}"
    )


def ensure_qwen_multimodal_processor(processor, model_path: str) -> None:
    required = ("tokenizer", "image_processor", "apply_chat_template")
    missing = [name for name in required if not hasattr(processor, name)]
    if not missing:
        return
    version = getattr(transformers, "__version__", "unknown")
    raise RuntimeError(
        "Qwen3-VL requires its official multimodal Processor, but "
        f"AutoProcessor returned {type(processor).__name__} (missing {missing}). "
        f"Installed transformers={version}. Activate the dedicated Qwen environment "
        "with transformers>=4.57.0; model path: " + model_path
    )


def qwen_tokenizer(processor):
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class QwenReferDataset(OriginalMaskReferDataset):
    """Original masks plus Qwen's official multimodal processor input."""

    def __init__(self, data_args: DataArguments, processor, qwen_image_size: int):
        super().__init__(data_args)
        self.processor = processor
        self.qwen_image_size = int(qwen_image_size)
        if self.qwen_image_size <= 0 or self.qwen_image_size % 32:
            raise ValueError("qwen_image_size must be a positive multiple of 32 for Qwen3-VL.")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        with Image.open(sample["image_path"]) as source:
            image = source.convert("RGB")
        image_processor = self.processor.image_processor
        mean = getattr(image_processor, "image_mean", (0.5, 0.5, 0.5))
        fill = tuple(int(max(0.0, min(1.0, float(value))) * 255.0) for value in mean)
        # Preserve SAM's top-left canvas convention before Qwen's square resize,
        # so visual evidence and the Teacher latent share one coordinate frame.
        canvas = sam_aligned_canvas(image, fill)
        qwen_image = canvas.resize((self.qwen_image_size, self.qwen_image_size), Image.BILINEAR)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": qwen_image},
                    {"type": "text", "text": qwen_prompt(sample["expression"])},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": " ".join(QWEN_LATENT_GRID_TOKENS)}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.processor(text=[text], images=[qwen_image], do_resize=False, return_tensors="pt", padding=False)
        sample["qwen_input_ids"] = encoded["input_ids"][0]
        sample["qwen_attention_mask"] = encoded["attention_mask"][0].bool()
        sample["qwen_pixel_values"] = encoded["pixel_values"]
        sample["qwen_image_grid_thw"] = encoded["image_grid_thw"]
        return sample


def _parse_sources(value: str):
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _default_split_by(dataset: str) -> str:
    return "umd" if dataset == "refcocog" else "unc"


def _resolve_reasonseg_split_dir(root: str, split: str) -> Path:
    root_path = Path(root)
    for candidate in (root_path / split, root_path / f"ReasonSeg_{split}", root_path):
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    raise FileNotFoundError(
        f"Could not find ReasonSeg '{split}' JSON files below {root_path}."
    )


def _find_reasonseg_image_path(annotation_path: Path, annotation: dict) -> Path:
    declared = str(annotation.get("imagePath", "")).strip()
    candidates = [annotation_path.parent / declared] if declared else []
    candidates.extend(annotation_path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image paired with ReasonSeg annotation: {annotation_path}")


def _reasonseg_prompt(expression: str, is_sentence, mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "official":
        sentence = is_sentence.strip().lower() in {"1", "true", "yes"} if isinstance(is_sentence, str) else bool(is_sentence)
        return f"{expression} Please output segmentation mask." if sentence else f"What is {expression} in this image? Please output segmentation mask."
    if mode == "question":
        return "Identify the image region that best answers the following question.\n" f"Question: {expression}\nPlease output the segmentation mask."
    raise ValueError("ReasonSeg training prompt mode must be official or question.")


def _reasonseg_polygon_mask(shapes, category: str, height: int, width: int, annotation_height, annotation_width):
    canvas = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(canvas)
    source_height = int(annotation_height) if annotation_height else height
    source_width = int(annotation_width) if annotation_width else width
    scale_x, scale_y = float(width) / source_width, float(height) / source_height
    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        is_ignore = "ignore" in label
        selected = is_ignore if category == "ignore" else (label != "flag" and not is_ignore)
        if not selected:
            continue
        points = shape.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        coordinates = [(float(point[0]) * scale_x, float(point[1]) * scale_y) for point in points]
        shape_type = str(shape.get("shape_type", "polygon")).strip().lower()
        if shape_type == "rectangle" and len(coordinates) >= 2:
            draw.rectangle((coordinates[0], coordinates[1]), fill=1)
        elif shape_type == "circle" and len(coordinates) >= 2:
            (cx, cy), (ex, ey) = coordinates[:2]
            radius = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=1)
        elif shape_type in {"line", "linestrip"}:
            draw.line(coordinates, fill=1, width=1)
        elif len(coordinates) >= 3:
            draw.polygon(coordinates, fill=1)
    return torch.from_numpy(np.asarray(canvas, dtype=bool))


class ReasonSegQwenDataset(Dataset):
    """ReasonSeg LabelMe annotations adapted to the llava6-3-8 trainer."""

    def __init__(self, data_args: DataArguments, processor, qwen_image_size: int):
        self.processor = processor
        self.qwen_image_size = int(qwen_image_size)
        self.prompt_mode = str(data_args.reasonseg_prompt_mode).strip().lower()
        if self.prompt_mode not in {"official", "question"}:
            raise ValueError("ReasonSeg training prompt mode must be official or question.")
        directory = _resolve_reasonseg_split_dir(data_args.reasonseg_root, data_args.train_split)
        records = []
        for annotation_path in sorted(directory.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            text = annotation.get("text", [])
            texts = [text] if isinstance(text, str) else [item for item in text if isinstance(item, str) and item.strip()]
            text_index = int(data_args.reasonseg_text_index)
            if not texts:
                raise ValueError(f"ReasonSeg annotation has no usable text: {annotation_path}")
            if not 0 <= text_index < len(texts):
                raise IndexError(f"reasonseg_text_index={text_index} is invalid for {annotation_path}")
            records.append((annotation_path, annotation, texts[text_index], _find_reasonseg_image_path(annotation_path, annotation)))
        expected = int(data_args.reasonseg_expected_count)
        if expected > 0 and len(records) != expected:
            raise ValueError(f"ReasonSeg sample count mismatch: found {len(records)}, expected {expected}.")
        if not records:
            raise RuntimeError(f"No ReasonSeg JSON annotations found in {directory}")
        self.records = records
        self.target_groups = [GROUP_SINGLE] * len(records)
        print(f"ReasonSeg training set: {len(records)} images, split={data_args.train_split}, prompt={self.prompt_mode}, directory={directory}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        annotation_path, annotation, expression, image_path = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        target = _reasonseg_polygon_mask(annotation.get("shapes", []), "target", image.height, image.width, annotation.get("imageHeight"), annotation.get("imageWidth"))
        ignore = _reasonseg_polygon_mask(annotation.get("shapes", []), "ignore", image.height, image.width, annotation.get("imageHeight"), annotation.get("imageWidth"))
        target = target & ~ignore
        mean = getattr(self.processor.image_processor, "image_mean", (0.5, 0.5, 0.5))
        fill = tuple(int(max(0.0, min(1.0, float(value))) * 255.0) for value in mean)
        qwen_image = sam_aligned_canvas(image, fill).resize((self.qwen_image_size, self.qwen_image_size), Image.BILINEAR)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": qwen_image},
                {"type": "text", "text": _reasonseg_prompt(expression, annotation.get("is_sentence", False), self.prompt_mode)},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": " ".join(QWEN_LATENT_GRID_TOKENS)}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.processor(text=[text], images=[qwen_image], do_resize=False, return_tensors="pt", padding=False)
        return {
            "sam_images": sam_image_input(image),
            "gt_masks": target[None].float(),
            "original_sizes": torch.tensor([image.height, image.width], dtype=torch.long),
            "sam_input_sizes": torch.tensor(sam_resize_size(image), dtype=torch.long),
            "target_groups": torch.tensor(GROUP_SINGLE, dtype=torch.long),
            "reference_key": annotation_path.stem, "expression": expression, "image_path": str(image_path),
            "qwen_input_ids": encoded["input_ids"][0], "qwen_attention_mask": encoded["attention_mask"][0].bool(),
            "qwen_pixel_values": encoded["pixel_values"], "qwen_image_grid_thw": encoded["image_grid_thw"],
        }


class MixedQwenReferDataset(Dataset):
    """Index-preserving concatenation of independently decoded RefCOCO sources."""

    def __init__(self, datasets: Sequence[QwenReferDataset], source_names: Sequence[str]):
        if not datasets:
            raise ValueError("MixedQwenReferDataset requires at least one source dataset.")
        if len(datasets) != len(source_names):
            raise ValueError("Dataset/source-name cardinalities must match.")
        self.datasets = list(datasets)
        self.source_names = list(source_names)
        self.cumulative_sizes = []
        self.target_groups = []
        self.source_ids = []
        running = 0
        for source_id, dataset in enumerate(self.datasets):
            running += len(dataset)
            self.cumulative_sizes.append(running)
            self.target_groups.extend(int(record[3]) for record in dataset.records)
            self.source_ids.extend([source_id] * len(dataset))
        self.records = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = bisect_right(self.cumulative_sizes, index)
        start = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        return self.datasets[source_index][index - start]


def build_train_dataset(data_args: DataArguments, processor, qwen_image_size: int) -> Dataset:
    source_names = _parse_sources(data_args.train_datasets)
    if not source_names:
        return QwenReferDataset(data_args, processor, qwen_image_size)
    if any(name in {"reasonseg", "reason_seg"} for name in source_names):
        if len(source_names) != 1:
            raise ValueError("ReasonSeg must be trained as a standalone source.")
        return ReasonSegQwenDataset(data_args, processor, qwen_image_size)
    split_bys = _parse_sources(data_args.train_split_bys) or [_default_split_by(name) for name in source_names]
    if len(split_bys) != len(source_names):
        raise ValueError("--train_split_bys must be empty or match --train_datasets in length.")
    caches = _parse_sources(data_args.hard_negative_cache)
    if len(caches) == 1:
        caches = caches * len(source_names)
    if caches and len(caches) != len(source_names):
        raise ValueError(
            "--hard_negative_cache must be one path or a pipe-separated path per --train_datasets source."
        )
    datasets = [
        QwenReferDataset(
            replace(
                data_args, refer_dataset=name, refer_split_by=split_by,
                train_datasets="", train_split_bys="",
                hard_negative_cache=(caches[index] if caches else ""),
            ),
            processor, qwen_image_size,
        )
        for index, (name, split_by) in enumerate(zip(source_names, split_bys))
    ]
    mixed = MixedQwenReferDataset(datasets, source_names)
    counts = ", ".join(f"{name}={len(dataset)}" for name, dataset in zip(source_names, datasets))
    print(f"llava6-3-4 mixed training set: total={len(mixed)} ({counts}).")
    return mixed


class QwenCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = qwen_tokenizer(processor)
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["qwen_input_ids"] for item in instances], batch_first=True, padding_value=self.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["qwen_attention_mask"].long() for item in instances], batch_first=True, padding_value=0,
        ).bool()
        return {
            **(
                {
                    "distractor_latents": torch.stack([item["distractor_latents"] for item in instances]),
                    "distractor_valid": torch.stack([item["distractor_valid"] for item in instances]),
                }
                if "distractor_latents" in instances[0] else {}
            ),
            "qwen_inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.cat([item["qwen_pixel_values"] for item in instances], dim=0),
                "image_grid_thw": torch.cat([item["qwen_image_grid_thw"] for item in instances], dim=0),
            },
            "sam_images": [item["sam_images"] for item in instances],
            "gt_masks": [item["gt_masks"] for item in instances],
            "original_sizes": torch.stack([item["original_sizes"] for item in instances]),
            "sam_input_sizes": torch.stack([item["sam_input_sizes"] for item in instances]),
            "target_groups": torch.stack([item["target_groups"] for item in instances]),
        }


def _target_groups(dataset: Dataset):
    groups = getattr(dataset, "target_groups", None)
    if groups is not None:
        return [int(group) for group in groups]
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError(f"Dataset {type(dataset).__name__} does not expose target groups for sampling.")
    return [int(record[3]) for record in records]


class RatioEmptyPositiveSampler(Sampler):
    """One pass over the positives with no-target expressions mixed in.

    llava6-3-1 forced a 1:1 empty/positive ratio, which is far from the true
    gRefCOCO distribution and pushed the model toward predicting nothing.
    """

    def __init__(self, dataset: Dataset, seed: int, positive_per_empty: int = 3):
        if int(positive_per_empty) <= 0:
            raise ValueError("positive_per_empty must be positive.")
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0
        self.ratio = int(positive_per_empty)
        groups = _target_groups(dataset)
        self.empty = [index for index, group in enumerate(groups) if group == GROUP_EMPTY]
        self.positive = [index for index, group in enumerate(groups) if group != GROUP_EMPTY]
        if not self.empty or not self.positive:
            raise RuntimeError("Ratio sampling needs both empty and positive expressions.")
        self.empty_count = max(1, len(self.positive) // self.ratio)

    def __len__(self):
        return len(self.positive) + self.empty_count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        positive = torch.tensor(self.positive, dtype=torch.long)[torch.randperm(len(self.positive), generator=generator)]
        empty = torch.tensor(self.empty, dtype=torch.long)[
            torch.randint(len(self.empty), (self.empty_count,), generator=generator)
        ]
        order, empty_cursor = [], 0
        for position, index in enumerate(positive.tolist()):
            order.append(index)
            if (position + 1) % self.ratio == 0 and empty_cursor < self.empty_count:
                order.append(int(empty[empty_cursor]))
                empty_cursor += 1
        order.extend(int(value) for value in empty[empty_cursor:])
        return iter(order)


class BalancedDatasetSampler(Sampler):
    """Sample mixed sources uniformly while keeping gRefCOCO's 3:1 ratio.

    The sampler keeps the epoch length equal to the concatenated dataset.  Each
    source receives an equal quota (difference at most one sample), and only a
    source that contains empty targets uses ``positive_per_empty`` internally.
    """

    def __init__(self, dataset: Dataset, seed: int, positive_per_empty: int = 3):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.ratio = int(positive_per_empty)
        if self.ratio <= 0:
            raise ValueError("positive_per_empty must be positive.")
        self.source_names = list(getattr(dataset, "source_names", []))
        self.source_ids = list(getattr(dataset, "source_ids", []))
        if len(self.source_names) <= 1 or len(self.source_ids) != len(dataset):
            raise ValueError("BalancedDatasetSampler requires a multi-source dataset.")
        self.groups = _target_groups(dataset)
        self.positive_by_source = []
        self.empty_by_source = []
        for source_id in range(len(self.source_names)):
            indices = [index for index, value in enumerate(self.source_ids) if value == source_id]
            self.positive_by_source.append(
                [index for index in indices if self.groups[index] != GROUP_EMPTY]
            )
            self.empty_by_source.append(
                [index for index in indices if self.groups[index] == GROUP_EMPTY]
            )
        self.base_quota, self.remainder = divmod(len(dataset), len(self.source_names))

    @staticmethod
    def _draw(pool, count, generator):
        if count <= 0 or not pool:
            return []
        if count <= len(pool):
            order = torch.randperm(len(pool), generator=generator)[:count].tolist()
            return [pool[position] for position in order]
        order = torch.randint(len(pool), (count,), generator=generator).tolist()
        return [pool[position] for position in order]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = []
        for source_id, (positive, empty) in enumerate(
            zip(self.positive_by_source, self.empty_by_source)
        ):
            quota = self.base_quota + (1 if source_id < self.remainder else 0)
            if empty and positive:
                empty_count = quota // (self.ratio + 1)
                positive_count = quota - empty_count
                order.extend(self._draw(positive, positive_count, generator))
                order.extend(self._draw(empty, empty_count, generator))
            elif positive:
                order.extend(self._draw(positive, quota, generator))
            else:
                # A source containing only empty targets is rare, but retaining
                # its quota is preferable to silently shortening the epoch.
                order.extend(self._draw(empty, quota, generator))
        shuffle = torch.randperm(len(order), generator=generator).tolist()
        return iter([order[position] for position in shuffle])


class Fp32FrozenLatentCallback(transformers.TrainerCallback):
    """Restore fp32 on the frozen latent target/decode path after DeepSpeed init."""

    def __init__(self, student: QwenLatentStudent):
        self.student = student

    def on_train_begin(self, args, state, control, **kwargs):
        restored = self.student.restore_fp32_modules()
        print(f"Restored fp32 precision for frozen modules: {restored}")



def training_summary(model, detail_window, grad_norms, model_args, data_args, state, clip) -> dict:
    """Turn the raw loss terms into the quantities that actually drive decisions.

    Every round of this project has been spent hand-converting Dice losses into
    IoU and comparing the latent error against the alpha-interpolation curve.
    Doing it here means the next run starts from numbers instead of arithmetic.
    """
    mean = {name: sum(values) / max(1, len(values)) for name, values in detail_window.items()}

    def iou_from_dice_loss(value):
        dice = 1.0 - float(value)
        return dice / max(1e-6, 2.0 - dice)

    summary = {
        "phase": model_args.training_phase,
        "global_step": int(state.global_step),
        "epoch": float(state.epoch or 0.0),
        "loss_terms_last_window": mean,
        "derived": {},
        "gradient": {},
        "learned_gates": model.calibration_scales(),
        "trainable_parameters": model.trainable_summary(),
        "settings": {
            "qwen_image_size": int(model_args.qwen_image_size),
            "latent_weight": float(model_args.latent_weight),
            "relative_latent_weight": float(model_args.relative_latent_weight),
            "head_bce_weight": float(model_args.head_bce_weight),
            "head_dice_weight": float(model_args.head_dice_weight),
            "mask_bce_weight": float(model_args.mask_bce_weight),
            "mask_dice_weight": float(model_args.mask_dice_weight),
            "phase_a_sam_mask_weight": float(model_args.phase_a_sam_mask_weight),
            "phase_a_sam_bce_weight": float(model_args.phase_a_sam_bce_weight),
            "phase_a_sam_dice_weight": float(model_args.phase_a_sam_dice_weight),
            "decoder_only": bool(model_args.decoder_only),
            "freeze_sam_decoder": bool(model_args.freeze_sam_decoder),
            "fp32_prompt_adapter": bool(model_args.fp32_prompt_adapter),
            "dedicated_prompt_scale_optimizer": bool(model_args.dedicated_prompt_scale_optimizer),
            "prompt_adapter_learning_rate": float(model_args.prompt_adapter_learning_rate),
            "prompt_adapter_scale_learning_rate": float(model_args.prompt_adapter_scale_learning_rate),
            "smooth_rank_weight": float(model_args.smooth_rank_weight),
            "smooth_rank_margin": float(model_args.smooth_rank_margin),
            "smooth_rank_temperature": float(model_args.smooth_rank_temperature),
            "hard_negative_cache": str(data_args.hard_negative_cache),
            "hard_negative_max": int(data_args.hard_negative_max),
            "dataset_sampling": str(data_args.dataset_sampling),
            "positive_per_empty": int(data_args.positive_per_empty),
            "train_datasets": data_args.train_datasets or data_args.refer_dataset,
            "train_subset_ratio": float(data_args.train_subset_ratio),
            "train_subset_seed": int(data_args.train_subset_seed),
        },
    }
    if "loss_mask_dice" in mean and mean["loss_mask_dice"] > 0:
        summary["derived"]["train_iou_sam_path"] = 100.0 * iou_from_dice_loss(mean["loss_mask_dice"])
    if "loss_head_dice" in mean and mean["loss_head_dice"] > 0:
        summary["derived"]["train_iou_image_free"] = 100.0 * iou_from_dice_loss(mean["loss_head_dice"])
    if "loss_latent_error" in mean:
        error = mean["loss_latent_error"]
        summary["derived"]["relative_latent_error"] = error
        # Measured on the llava6-3-2 checkpoint: IoU 80.57 at error 0.191 and
        # 86.58 at 0.143, i.e. -125.2 IoU per unit relative error.  It is a
        # frozen-decoder reference point, so Phase B is expected to beat it.
        summary["derived"]["alpha_curve_reference_iou"] = 80.57 - 125.2 * (error - 0.191)
        summary["derived"]["alpha_curve_note"] = (
            "frozen-decoder reference from grefcoco testA; a Phase-B model above "
            "this line means the decoder adaptation is paying off"
        )
    if "loss_smooth_rank_violation_rate" in mean:
        summary["derived"]["smooth_rank_violation_rate"] = mean["loss_smooth_rank_violation_rate"]
    if grad_norms:
        ordered = sorted(grad_norms)
        summary["gradient"] = {
            "median_norm": ordered[len(ordered) // 2],
            "p90_norm": ordered[int(0.9 * (len(ordered) - 1))],
            "max_norm": ordered[-1],
            "clip_threshold": float(clip),
            "note": "if the median is far above the clip threshold, every step is being rescaled",
        }
    # Contribution of each term to the total, which is what actually decides
    # where the optimizer spends its capacity.
    if model_args.training_phase == "latent_warmup":
        weights = {
            "loss_latent": float(model_args.latent_weight),
            "loss_latent_error": float(model_args.relative_latent_weight),
            "loss_head_bce": float(model_args.head_bce_weight),
            "loss_head_dice": float(model_args.head_dice_weight),
            "loss_mask_bce": float(model_args.phase_a_sam_mask_weight * model_args.phase_a_sam_bce_weight),
            "loss_mask_dice": float(model_args.phase_a_sam_mask_weight * model_args.phase_a_sam_dice_weight),
        }
    else:
        weights = {
            "loss_latent": float(model_args.latent_weight),
            "loss_latent_error": float(model_args.relative_latent_weight),
            "loss_mask_bce": float(model_args.mask_bce_weight),
            "loss_mask_dice": float(model_args.mask_dice_weight),
            "loss_smooth_rank": float(model_args.smooth_rank_weight),
        }
    contributions = {name: weights[name] * mean.get(name, 0.0) for name in weights}
    total = sum(contributions.values())
    summary["loss_contribution_share"] = {
        name: (100.0 * value / total if total > 0 else 0.0)
        for name, value in sorted(contributions.items(), key=lambda item: -item[1])
    }
    return summary


class QwenLatentTrainer(transformers.Trainer):
    def __init__(self, *args, processor, model_args: ModelArguments, data_args: DataArguments, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor, self.model_args, self.data_args = processor, model_args, data_args
        self._detail_sums = {}
        self._detail_count = 0
        self._detail_window = {}
        self._grad_norms = []

    def log(self, logs, *args, **kwargs):
        # HF already computes the pre-clip gradient norm; keep it so the summary
        # can say whether the clip threshold is actually binding.
        norm = logs.get("grad_norm")
        if norm is not None:
            try:
                self._grad_norms.append(float(norm))
                self._grad_norms = self._grad_norms[-500:]
            except (TypeError, ValueError):
                pass
        return super().log(logs, *args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        groups = _target_groups(dataset)
        has_empty = any(group == GROUP_EMPTY for group in groups)
        has_positive = any(group != GROUP_EMPTY for group in groups)
        ratio = int(self.data_args.positive_per_empty)
        sampling = str(self.data_args.dataset_sampling).strip().lower()
        source_names = getattr(dataset, "source_names", [])
        if sampling not in {"balanced", "proportional"}:
            raise ValueError("dataset_sampling must be 'balanced' or 'proportional'.")
        if sampling == "balanced" and len(source_names) > 1:
            return BalancedDatasetSampler(
                dataset, seed=int(self.args.seed), positive_per_empty=ratio
            )
        if ratio > 0 and has_empty and has_positive:
            return RatioEmptyPositiveSampler(dataset, seed=int(self.args.seed), positive_per_empty=ratio)
        return super()._get_train_sampler(train_dataset)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_student(os.path.join(output_dir, "qwen_latent_student.pt"))
        try:
            summary = training_summary(
                model, self._detail_window, self._grad_norms, self.model_args, self.data_args,
                self.state, self.args.max_grad_norm,
            )
            with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print("Training summary:", json.dumps(summary.get("derived", {}), indent=2))
            print("Loss contribution share:", json.dumps(summary.get("loss_contribution_share", {}), indent=2))
        except Exception as error:  # a summary must never cost you a checkpoint
            print(f"Could not write training_summary.json: {error}")
        self.processor.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "qwen_latent_config.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "format": model.FORMAT,
                "model": asdict(self.model_args),
                "calibration": model.calibration_scales(),
            }, handle, indent=2)

    def _load_from_checkpoint(self, *args, **kwargs):
        """Load the custom Student weights used by every Trainer checkpoint.

        ``QwenLatentStudent`` is not a ``PreTrainedModel`` and therefore its
        compact ``qwen_latent_student.pt`` file is written by ``_save`` rather
        than by Transformers' standard ``pytorch_model.bin`` path.  The
        optimizer, scheduler, and trainer state are still restored by the
        parent Trainer after this hook returns.
        """
        # Transformers has used both ``(resume_from_checkpoint, model=None)``
        # and ``(model, resume_from_checkpoint)`` in Trainer releases used by
        # this repository.  Accept either form so Qwen3-VL environments can
        # resume without a version-specific argument error.
        model = kwargs.pop("model", None)
        resume_from_checkpoint = kwargs.pop("resume_from_checkpoint", None)
        if len(args) == 2:
            if isinstance(args[0], (str, os.PathLike)):
                resume_from_checkpoint, model = args[0], args[1]
            else:
                model, resume_from_checkpoint = args[0], args[1]
        elif len(args) == 1:
            if isinstance(args[0], (str, os.PathLike)):
                resume_from_checkpoint = args[0]
            else:
                model = args[0]
        if resume_from_checkpoint is None:
            raise TypeError("_load_from_checkpoint requires a resume checkpoint directory")
        model = self.model if model is None else model
        target = model.module if hasattr(model, "module") else model
        student_path = os.path.join(resume_from_checkpoint, "qwen_latent_student.pt")
        if not os.path.isfile(student_path):
            raise FileNotFoundError(
                f"Qwen latent resume checkpoint is missing: {student_path}. "
                "Use a checkpoint-N directory created by this Trainer, not only the final output file."
            )
        target.load_student(student_path)
        print(f"Resumed Student weights from {student_path}; Trainer will restore optimizer/scheduler/state.")

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        raw = self.model.module if hasattr(self.model, "module") else self.model
        qwen_names = raw.qwen_parameter_names()
        grid_ids = raw.grid_embedding_parameter_ids()
        adapter_names = raw.adapter_parameter_names()
        scale_names = raw.calibrator_scale_parameter_names()
        prompt_scale_names = raw.prompt_adapter_scale_parameter_names() if self.model_args.dedicated_prompt_scale_optimizer else set()
        adapter_names = adapter_names - scale_names
        prompt_names = raw.prompt_adapter_parameter_names()
        prompt_names = prompt_names - prompt_scale_names
        decoder_names = raw.decoder_parameter_names()
        qwen, adapter, scales, prompt_scales, prompt, decoder = [], [], [], [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            normalized = name.removeprefix("module.")
            if normalized in qwen_names or id(parameter) in grid_ids:
                qwen.append(parameter)
            elif normalized in scale_names:
                scales.append(parameter)
            elif normalized in prompt_scale_names:
                prompt_scales.append(parameter)
            elif normalized in adapter_names:
                adapter.append(parameter)
            elif normalized in prompt_names:
                prompt.append(parameter)
            elif normalized in decoder_names:
                decoder.append(parameter)
            else:
                raise RuntimeError(f"Unexpected trainable Student parameter: {normalized}")
        groups = [
            {"params": qwen, "lr": float(self.model_args.qwen_lora_learning_rate), "weight_decay": self.args.weight_decay},
            {"params": adapter, "lr": float(self.model_args.adapter_learning_rate), "weight_decay": self.args.weight_decay},
            {"params": scales, "lr": float(self.model_args.calibrator_scale_learning_rate), "weight_decay": 0.0},
            {"params": prompt_scales, "lr": float(self.model_args.prompt_adapter_scale_learning_rate), "weight_decay": 0.0},
            {"params": prompt, "lr": float(self.model_args.prompt_adapter_learning_rate), "weight_decay": self.args.weight_decay},
            {"params": decoder, "lr": float(self.model_args.decoder_half_learning_rate), "weight_decay": self.args.weight_decay},
        ]
        groups = [group for group in groups if group["params"]]
        print(
            "Optimizer groups:",
            [(float(group["lr"]), sum(parameter.numel() for parameter in group["params"])) for group in groups],
        )
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task_adaptation = self.model_args.training_phase == "task_adaptation"
        output = model(**inputs, compute_mask_losses=task_adaptation)
        loss = output["loss"]
        if not torch.isfinite(loss).all() or float(loss.detach().float()) > float(self.args.dense_mask_abort_loss):
            terms = {key: float(value.detach().float()) for key, value in output.items() if key.startswith("loss_")}
            raise FloatingPointError(f"llava6-3-8 loss diverged: {float(loss.detach().float())}; {terms}")
        if model.training and bool(self.args.dense_mask_log_details):
            # compute_loss runs once per micro-batch; average over the
            # accumulation window so one optimizer step logs one line.
            values = {key: float(value.detach().float().cpu()) for key, value in output.items() if key != "loss"}
            values["loss_total"] = float(loss.detach().float().cpu())
            for name, value in values.items():
                self._detail_sums[name] = self._detail_sums.get(name, 0.0) + value
            self._detail_count += 1
            at_boundary = bool(getattr(self.accelerator, "sync_gradients", True))
            due = (int(self.state.global_step) + 1) % max(1, int(self.args.logging_steps)) == 0
            if at_boundary and due:
                for name, value in self._detail_sums.items():
                    self._detail_window.setdefault(name, []).append(value / max(1, self._detail_count))
                    self._detail_window[name] = self._detail_window[name][-25:]
                if self.is_world_process_zero() and self._detail_count:
                    raw = model.module if hasattr(model, "module") else model
                    record = {f"student/{name}": value / self._detail_count for name, value in self._detail_sums.items()}
                    # Log whatever gates the model currently exposes.  Naming
                    # them individually here is what broke training when the
                    # image-free calibrator was removed, four minutes in.
                    record.update({
                        f"student/{name}": value
                        for name, value in raw.calibration_scales().items()
                        if isinstance(value, (int, float))
                    })
                    self.log(record)
                self._detail_sums.clear()
                self._detail_count = 0
        return (loss, output) if return_outputs else loss


def build_qwen(model_args: ModelArguments):
    if not os.path.isdir(model_args.qwen_model_path):
        raise FileNotFoundError(f"Qwen3-VL checkpoint directory not found: {model_args.qwen_model_path}")
    processor = transformers.AutoProcessor.from_pretrained(model_args.qwen_model_path)
    ensure_qwen_multimodal_processor(processor, model_args.qwen_model_path)
    tokenizer = qwen_tokenizer(processor)
    tokenizer.add_special_tokens({"additional_special_tokens": list(QWEN_LATENT_GRID_TOKENS)})
    load_kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if bool(model_args.use_flash_attention_2):
        load_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        qwen = transformers.AutoModelForImageTextToText.from_pretrained(model_args.qwen_model_path, **load_kwargs)
    except Exception as exc:
        if not bool(model_args.use_flash_attention_2):
            raise
        print(f"FlashAttention-2 was unavailable ({exc}); retrying with the default attention implementation.")
        load_kwargs.pop("attn_implementation", None)
        qwen = transformers.AutoModelForImageTextToText.from_pretrained(model_args.qwen_model_path, **load_kwargs)
    qwen.resize_token_embeddings(len(tokenizer))
    grid_ids = tokenizer.convert_tokens_to_ids(list(QWEN_LATENT_GRID_TOKENS))
    if len(set(grid_ids)) != len(grid_ids) or any(token_id < 0 for token_id in grid_ids):
        raise RuntimeError("The Qwen tokenizer did not preserve unique latent grid tokens.")
    return processor, qwen, grid_ids


def parse_readout_layers(value: str):
    """Parse "-1,-9,-17".

    argparse treats a bare value starting with '-' as another option, so the
    shell scripts pass it glued to the flag ("--readout_layers=-1,-9,-17").
    An "n" prefix ("n1,n9,n17") is accepted as an escape hatch for shells or
    launchers that mangle the '=' form.
    """
    layers = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(-int(item[1:]) if item[0] in "nN" else int(item))
    if not layers:
        raise ValueError("--readout_layers must list at least one layer index.")
    return tuple(layers)


def build_student(model_args: ModelArguments):
    """Shared construction path for training, evaluation and diagnostics."""
    processor, qwen, grid_ids = build_qwen(model_args)
    teacher = MaskLatentTeacher(model_args.sam_path, model_args.sam_model_type)
    if model_args.teacher_checkpoint:
        teacher.load_teacher(model_args.teacher_checkpoint)
    else:
        print(
            "No Stage-1 Teacher checkpoint supplied; keeping the task-specific "
            "Teacher prompt/decoder modules at base initialization."
        )
    student = QwenLatentStudent(
        qwen, grid_ids, teacher,
        lora_r=model_args.qwen_lora_r,
        lora_alpha=model_args.qwen_lora_alpha,
        lora_dropout=model_args.qwen_lora_dropout,
        lora_target_mlp=bool(model_args.qwen_lora_target_mlp),
        latent_width=model_args.latent_width,
        latent_depth=model_args.latent_depth,
        text_cross_attention_heads=model_args.text_cross_attention_heads,
        readout_layers=parse_readout_layers(model_args.readout_layers),
        latent_weight=model_args.latent_weight,
        relative_latent_weight=model_args.relative_latent_weight,
        head_bce_weight=model_args.head_bce_weight,
        head_dice_weight=model_args.head_dice_weight,
        mask_bce_weight=model_args.mask_bce_weight,
        mask_dice_weight=model_args.mask_dice_weight,
        smooth_rank_weight=model_args.smooth_rank_weight,
        smooth_rank_margin=model_args.smooth_rank_margin,
        smooth_rank_temperature=model_args.smooth_rank_temperature,
        empty_loss_weight=model_args.empty_loss_weight,
        mask_threshold=model_args.mask_threshold,
        prompt_calibrator_max_scale=model_args.prompt_calibrator_max_scale,
        image_prompt_calibrator_width=model_args.image_prompt_calibrator_width,
        image_prompt_calibrator_depth=model_args.image_prompt_calibrator_depth,
        image_prompt_calibrator_initial_scale=model_args.image_prompt_calibrator_initial_scale,
        phase_a_sam_mask_weight=model_args.phase_a_sam_mask_weight,
        phase_a_sam_bce_weight=model_args.phase_a_sam_bce_weight,
        phase_a_sam_dice_weight=model_args.phase_a_sam_dice_weight,
        decoder_only=model_args.decoder_only,
        freeze_sam_decoder=model_args.freeze_sam_decoder,
        sam_decoder_tuning=model_args.sam_decoder_tuning,
        fp32_prompt_adapter=model_args.fp32_prompt_adapter,
        training_phase=model_args.training_phase,
    )
    return processor, student


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.stage != "student":
        raise ValueError("llava6-3-8 Stage 2 supports only --stage student; Stage 1 has its own entry point.")
    if model_args.training_phase not in {"latent_warmup", "task_adaptation"}:
        raise ValueError("--training_phase must be latent_warmup or task_adaptation.")
    if model_args.sam_decoder_tuning not in {"student_b", "student_b_full"}:
        raise ValueError("--sam_decoder_tuning must be student_b or student_b_full.")
    if not os.path.isfile(model_args.sam_path):
        raise FileNotFoundError("Stage 2 requires a valid --sam_path.")
    if model_args.teacher_checkpoint and not os.path.isfile(model_args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {model_args.teacher_checkpoint}")
    processor, model = build_student(model_args)
    model = model.to(dtype=torch.bfloat16)
    model.teacher.install_trainable_gradient_guards()
    if model_args.student_checkpoint:
        if not os.path.isfile(model_args.student_checkpoint):
            raise FileNotFoundError(f"Student checkpoint not found: {model_args.student_checkpoint}")
        model.load_student(model_args.student_checkpoint)
        print(f"Initialized the Student from {model_args.student_checkpoint}; optimizer and scheduler restart fresh.")
    data_args.load_sam_images = (
        model_args.training_phase == "task_adaptation"
        or model_args.phase_a_sam_mask_weight > 0.0
    )
    dataset = build_train_dataset(data_args, processor, model_args.qwen_image_size)
    print("llava6-3-8 trainable parameters:", model.trainable_summary())
    print(f"SAM canvases loaded by the dataloader: {data_args.load_sam_images}")
    callbacks = [Fp32FrozenLatentCallback(model)] if bool(model_args.fp32_frozen_latent_modules) else None
    trainer = QwenLatentTrainer(
        model=model, args=training_args, train_dataset=dataset, data_collator=QwenCollator(processor),
        processor=processor, model_args=model_args, data_args=data_args, callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint or None)
    if trainer.is_world_process_zero():
        trainer._save(training_args.output_dir)


if __name__ == "__main__":
    main()





