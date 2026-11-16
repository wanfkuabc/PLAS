"""Zero-shot ReasonSeg evaluation for the isolated Qwen dense-latent branch.

ReasonSeg stores one image and one LabelMe-style JSON annotation per example.
Its ``ignore`` polygons are excluded from both prediction and target before IoU
is computed, matching the public LISA/ReasonSeg evaluation convention.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from PLAS.model.qwen_dense_latent_refseg import QWEN_LATENT_GRID_TOKENS
from PLAS.train.qwen_sam_geometry import sam_aligned_canvas, sam_image_input, sam_resize_size
from PLAS.train.train_qwen_dense_latent_refseg import ModelArguments, QwenCollator, build_student, qwen_prompt


@dataclass
class ReasonSegArguments:
    reasonseg_root: str = "./PLAS/datasets/ReasonSeg"
    split: str = "val"
    reasonseg_text_index: int = 0
    # ReasonSeg should use its public evaluation-style prompt by default.
    # ``question`` remains available for question-form expressions, while the
    # RefCOCO-oriented ``current`` prompt must be selected explicitly.
    reasonseg_prompt_mode: str = "official"
    output_dir: str = "./PLAS/outputs/reasonseg_evaluation"
    threshold: float = 0.55
    save_visuals: bool = True
    visual_max_samples: int = 256
    local_rank: int = -1


def initialize_distributed():
    if "RANK" in os.environ and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world, device


def resolve_split_dir(root: str, split: str) -> Path:
    root_path = Path(root)
    candidates = (root_path / split, root_path / f"ReasonSeg_{split}", root_path)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    raise FileNotFoundError(
        f"Could not find ReasonSeg '{split}' JSON files below {root_path}. Expected "
        f"{root_path / split}/<sample>.json paired with <sample>.jpg."
    )


def find_image_path(annotation_path: Path, annotation: dict) -> Path:
    declared = str(annotation.get("imagePath", "")).strip()
    candidates = [annotation_path.parent / declared] if declared else []
    candidates.extend(annotation_path.with_suffix(extension) for extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image paired with ReasonSeg annotation: {annotation_path}")


def reasonseg_prompt(expression: str, is_sentence: bool, mode: str) -> str:
    """Build a fixed ReasonSeg prompt without changing the checkpoint.

    ``current`` preserves the RefCOCO-oriented PLAS template.  ``official``
    follows LISA's public ReasonSeg validation formatting.  ``question`` keeps
    the full question and removes the visible-evidence restriction so that
    commonsense/world-knowledge queries are not explicitly suppressed.
    """
    mode = str(mode).strip().lower()
    if mode == "current":
        return qwen_prompt(expression)
    if mode == "official":
        sentence_flag = (
            is_sentence.strip().lower() in {"1", "true", "yes"}
            if isinstance(is_sentence, str)
            else bool(is_sentence)
        )
        if sentence_flag:
            return f"{expression} Please output segmentation mask."
        return f"What is {expression} in this image? Please output segmentation mask."
    if mode == "question":
        return (
            "Identify the image region that best answers the following question.\n"
            f"Question: {expression}\n"
            "Please output the segmentation mask."
        )
    raise ValueError(
        f"Unsupported reasonseg_prompt_mode={mode!r}; use current, official, or question."
    )


def polygon_mask(
    shapes: list[dict], category: str, height: int, width: int, annotation_height: int | None, annotation_width: int | None,
) -> np.ndarray:
    canvas = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(canvas)
    source_height = int(annotation_height) if annotation_height else height
    source_width = int(annotation_width) if annotation_width else width
    scale_x, scale_y = float(width) / source_width, float(height) / source_height
    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        # Match LISA's public parser: deprecated ``flag`` shapes are skipped;
        # every non-ignore shape is a valid target, not only literal "target".
        is_ignore = "ignore" in label
        is_selected = is_ignore if category == "ignore" else (label != "flag" and not is_ignore)
        if not is_selected:
            continue
        points = shape.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        coordinates = [(float(point[0]) * scale_x, float(point[1]) * scale_y) for point in points]
        shape_type = str(shape.get("shape_type", "polygon")).strip().lower()
        if shape_type == "rectangle" and len(coordinates) >= 2:
            draw.rectangle((coordinates[0], coordinates[1]), fill=1)
        elif shape_type == "circle" and len(coordinates) >= 2:
            (center_x, center_y), (edge_x, edge_y) = coordinates[:2]
            radius = ((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2) ** 0.5
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=1)
        elif shape_type in {"line", "linestrip"}:
            draw.line(coordinates, fill=1, width=1)
        elif len(coordinates) >= 3:
            draw.polygon(coordinates, fill=1)
    return np.asarray(canvas, dtype=np.uint8).astype(bool)


class ReasonSegDataset(Dataset):
    """One deterministic instruction (index zero by default) per ReasonSeg image."""

    def __init__(self, args: ReasonSegArguments, processor, qwen_image_size: int):
        if args.split not in {"val", "test"}:
            raise ValueError("ReasonSeg supports --split val or --split test.")
        self.args, self.processor = args, processor
        self.prompt_mode = str(args.reasonseg_prompt_mode).strip().lower()
        if self.prompt_mode not in {"current", "official", "question"}:
            raise ValueError(
                "reasonseg_prompt_mode must be current, official, or question."
            )
        self.qwen_image_size = int(qwen_image_size)
        if self.qwen_image_size <= 0 or self.qwen_image_size % 32:
            raise ValueError("qwen_image_size must be a positive multiple of 32 for Qwen3-VL.")
        directory = resolve_split_dir(args.reasonseg_root, args.split)
        self.records = []
        for annotation_path in sorted(directory.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            text = annotation.get("text", [])
            texts = [text] if isinstance(text, str) else [item for item in text if isinstance(item, str) and item.strip()]
            if not texts:
                raise ValueError(f"ReasonSeg annotation has no usable text: {annotation_path}")
            if not 0 <= int(args.reasonseg_text_index) < len(texts):
                raise IndexError(
                    f"--reasonseg_text_index={args.reasonseg_text_index} is invalid for {annotation_path}; "
                    f"available instructions={len(texts)}."
                )
            self.records.append((annotation_path, annotation, texts[int(args.reasonseg_text_index)], find_image_path(annotation_path, annotation)))
        if not self.records:
            raise RuntimeError(f"No ReasonSeg JSON annotations found in {directory}")
        print(f"ReasonSeg zero-shot set: {len(self.records)} images, split={args.split}, directory={directory}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        annotation_path, annotation, expression, image_path = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        annotation_height = annotation.get("imageHeight")
        annotation_width = annotation.get("imageWidth")
        is_sentence = annotation.get("is_sentence", False)
        target = polygon_mask(
            annotation.get("shapes", []), "target", image.height, image.width, annotation_height, annotation_width,
        )
        ignore = polygon_mask(
            annotation.get("shapes", []), "ignore", image.height, image.width, annotation_height, annotation_width,
        )
        image_processor = self.processor.image_processor
        mean = getattr(image_processor, "image_mean", (0.5, 0.5, 0.5))
        fill = tuple(int(max(0.0, min(1.0, float(value))) * 255.0) for value in mean)
        qwen_image = sam_aligned_canvas(image, fill).resize((self.qwen_image_size, self.qwen_image_size), Image.BILINEAR)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": qwen_image}, {"type": "text", "text": reasonseg_prompt(expression, is_sentence, self.prompt_mode)}]},
            {"role": "assistant", "content": [{"type": "text", "text": " ".join(QWEN_LATENT_GRID_TOKENS)}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.processor(text=[text], images=[qwen_image], do_resize=False, return_tensors="pt", padding=False)
        return {
            "sam_images": sam_image_input(image),
            "gt_masks": torch.from_numpy(target[None].astype(np.float32)),
            "ignore_mask": torch.from_numpy(ignore[None]),
            "original_sizes": torch.tensor([image.height, image.width], dtype=torch.long),
            "sam_input_sizes": torch.tensor(sam_resize_size(image), dtype=torch.long),
            "target_groups": torch.tensor(0, dtype=torch.long),
            "reference_key": annotation_path.stem,
            "expression": expression,
            "image_path": str(image_path),
            "qwen_input_ids": encoded["input_ids"][0],
            "qwen_attention_mask": encoded["attention_mask"][0].bool(),
            "qwen_pixel_values": encoded["pixel_values"],
            "qwen_image_grid_thw": encoded["image_grid_thw"],
        }


def save_visual(output_dir, index, rank, sample, prediction, target, ignore):
    directory = Path(output_dir) / "visuals" / f"rank_{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample["reference_key"])[:96]
    stem = f"{index:06d}_{key}"
    with Image.open(sample["image_path"]) as source:
        image = source.convert("RGB")
    predicted = np.squeeze(prediction.detach().cpu().numpy()).astype(bool)
    ground_truth = np.squeeze(target.detach().cpu().numpy()).astype(bool)
    ignored = np.squeeze(ignore.detach().cpu().numpy()).astype(bool)
    Image.fromarray((predicted * 255).astype(np.uint8)).save(directory / f"{stem}_pred.png")
    Image.fromarray((ground_truth * 255).astype(np.uint8)).save(directory / f"{stem}_gt.png")
    image.save(directory / f"{stem}_image.jpg", quality=95)
    overlay = np.asarray(image).astype(np.float32)
    overlay[ground_truth] = 0.55 * overlay[ground_truth] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
    overlay[predicted] = 0.55 * overlay[predicted] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    overlay[ignored] = 0.55 * overlay[ignored] + 0.45 * np.array([255, 255, 0], dtype=np.float32)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(directory / f"{stem}_overlay.jpg", quality=95)
    with (directory / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump({"reference_key": sample["reference_key"], "expression": sample["expression"], "overlay": "green=GT, red=prediction, yellow=ignored"}, handle, ensure_ascii=False, indent=2)


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((ModelArguments, ReasonSegArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()
    rank, world, device = initialize_distributed()
    if not os.path.isfile(model_args.teacher_checkpoint) or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("ReasonSeg evaluation requires --teacher_checkpoint and --student_checkpoint.")
    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()
    dataset = ReasonSegDataset(eval_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)
    # image IoU sum, image count, global intersection, global union,
    # no-object correct count, no-object count
    stats = torch.zeros(6, device=device, dtype=torch.float64)
    for index in range(rank, len(dataset), world):
        sample = dataset[index]
        batch = collator([sample])
        qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
        logits, _ = model.predict(
            qwen_inputs, batch["sam_images"], batch["sam_input_sizes"].to(device), batch["original_sizes"].to(device),
        )
        logits = logits[0]
        prediction = logits.sigmoid() >= float(eval_args.threshold)
        target = batch["gt_masks"][0].to(device=device, dtype=torch.bool)
        ignore = sample["ignore_mask"].to(device=device, dtype=torch.bool)
        prediction = prediction & ~ignore
        target = target & ~ignore
        intersection = (prediction & target).sum(dtype=torch.float64)
        union = (prediction | target).sum(dtype=torch.float64)
        # Follow the public LISA evaluator: a zero-union ReasonSeg annotation
        # is a no-object example and receives IoU 1 only when prediction is also empty.
        stats[0] += torch.where(union == 0, torch.ones_like(union), intersection / union)
        stats[1] += 1
        stats[2] += intersection
        stats[3] += union
        target_is_empty = not bool(target.any())
        if target_is_empty:
            stats[4] += float(not bool(prediction.any()))
            stats[5] += 1
        if eval_args.save_visuals and (eval_args.visual_max_samples < 0 or index < eval_args.visual_max_samples):
            save_visual(eval_args.output_dir, index, rank, sample, prediction, target, ignore)
            if index % 20 == 0:
                print(f"[rank {rank}] saved visualization through index {index}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    if rank == 0:
        metrics = {
            "split": eval_args.split,
            "gIoU": float((100.0 * stats[0] / stats[1].clamp_min(1.0)).item()),
            "cIoU": float((100.0 * stats[2] / stats[3].clamp_min(1.0)).item()),
            "threshold": float(eval_args.threshold),
            "count": int(stats[1].item()),
            "ignore_policy": "ignore polygons excluded from both prediction and target",
            "text_index": int(eval_args.reasonseg_text_index),
            "prompt_mode": str(eval_args.reasonseg_prompt_mode),
            "diagnostics": {
                "no_target_count": int(stats[5].item()),
                "no_target_accuracy": float((100.0 * stats[4] / stats[5].clamp_min(1.0)).item()),
            },
        }
        os.makedirs(eval_args.output_dir, exist_ok=True)
        path = Path(eval_args.output_dir) / f"{eval_args.split}_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print(f"Saved ReasonSeg zero-shot evaluation: {path}")


if __name__ == "__main__":
    main()





