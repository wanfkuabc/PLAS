"""Evaluate the llava6-3-8 Stage-I Teacher reconstruction ceiling.

The evaluation sweeps relative latent perturbations and measures both the
image-free mask head and the image-conditioned SAM path.  It also saves
original-image visualizations for the head output at selected noise levels.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw

# Also support ``python PLAS/eval/eval_mask_latent_teacher.py`` without
# requiring the caller to export PYTHONPATH first.  The shell wrapper still
# sets it explicitly for DeepSpeed workers.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PLAS.model.mask_latent_teacher import (
    MaskLatentTeacher,
    mask_bce_dice,
    perturb_latent,
    relative_latent_error,
)
from PLAS.train.qwen_refseg_data import GROUP_EMPTY, OriginalMaskReferDataset
from PLAS.train.train_mask_latent_teacher import DataArguments, TeacherCollator


@dataclass
class TeacherEvalArguments:
    sam_path: str = "./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    teacher_checkpoint: str = ""
    output_dir: str = "./PLAS/outputs/teacher_evaluation"
    split: str = "val"
    threshold: float = 0.55
    max_samples: int = 2000
    noise_sigmas: str = "0.0,0.1,0.2,0.3,0.4,0.5"
    noise_repeats: int = 1
    visual_sigmas: str = "0.4"
    visual_max_samples: int = 32
    save_visuals: bool = True
    include_empty: bool = False
    seed: int = 3407
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


def parse_floats(value: str) -> list[float]:
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            number = float(item)
            if number < 0.0:
                raise ValueError(f"Noise sigma must be non-negative, got {number}.")
            result.append(number)
    if not result:
        raise ValueError("At least one noise sigma is required.")
    return result


def mask_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float):
    target = target.to(device=logits.device, dtype=torch.float32)
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    bce, dice = mask_bce_dice(logits, target)
    prediction = logits.sigmoid() >= float(threshold)
    truth = target >= 0.5
    intersection = (prediction & truth).sum(dtype=torch.float64)
    union = (prediction | truth).sum(dtype=torch.float64)
    sample_iou = intersection / union.clamp_min(1.0)
    return intersection, union, sample_iou, bce.mean(), dice.mean()


def _rgb_mask(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def _overlay(image: Image.Image, gt: np.ndarray, prediction: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = base.copy()
    gt_mask = gt.astype(bool)
    pred_mask = prediction.astype(bool)
    result[gt_mask] = 0.55 * result[gt_mask] + 0.45 * np.asarray([0, 255, 0], dtype=np.float32)
    result[pred_mask] = 0.55 * result[pred_mask] + 0.45 * np.asarray([255, 0, 0], dtype=np.float32)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8), mode="RGB")


def save_head_visual(
    output_dir: str,
    index: int,
    rank: int,
    sigma: float,
    sample: dict,
    gt: np.ndarray,
    head_clean: np.ndarray,
    mask_clean: np.ndarray,
    head_noisy: np.ndarray,
    mask_noisy: np.ndarray,
    metrics: dict,
) -> None:
    directory = os.path.join(output_dir, "visuals", f"rank_{rank}", f"sigma_{sigma:.3f}")
    os.makedirs(directory, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample.get("reference_key", index)))[:96]
    stem = f"{index:06d}_{key}"
    with Image.open(sample["image_path"]) as source:
        image = source.convert("RGB")
    clean = head_clean.astype(bool)
    clean_mask = mask_clean.astype(bool)
    noisy = head_noisy.astype(bool)
    noisy_mask = mask_noisy.astype(bool)
    image.save(os.path.join(directory, f"{stem}_image.jpg"), quality=95)
    Image.fromarray((gt.astype(np.uint8) * 255), mode="L").save(os.path.join(directory, f"{stem}_gt.png"))
    Image.fromarray((clean.astype(np.uint8) * 255), mode="L").save(os.path.join(directory, f"{stem}_head_clean.png"))
    Image.fromarray((clean_mask.astype(np.uint8) * 255), mode="L").save(os.path.join(directory, f"{stem}_mask_clean.png"))
    Image.fromarray((noisy.astype(np.uint8) * 255), mode="L").save(os.path.join(directory, f"{stem}_head_noisy.png"))
    Image.fromarray((noisy_mask.astype(np.uint8) * 255), mode="L").save(os.path.join(directory, f"{stem}_mask_noisy.png"))

    panels = [
        image,
        _rgb_mask(gt, (0, 180, 0)),
        _rgb_mask(clean, (255, 180, 0)),
        _rgb_mask(clean_mask, (0, 120, 255)),
        _rgb_mask(noisy, (255, 0, 0)),
        _rgb_mask(noisy_mask, (180, 0, 255)),
    ]
    panel_size = 256
    panel = Image.new("RGB", (panel_size * len(panels), panel_size + 24), "white")
    labels = (
        "image", "GT", "head clean (sigma=0)", "mask clean (sigma=0)",
        f"head noisy (sigma={sigma:g})", f"mask noisy (sigma={sigma:g})",
    )
    draw = ImageDraw.Draw(panel)
    for position, (part, label) in enumerate(zip(panels, labels)):
        panel.paste(part.resize((panel_size, panel_size)), (position * panel_size, 24))
        draw.text((position * panel_size + 4, 4), label, fill="black")
    panel_path = os.path.join(directory, f"{stem}_head_panel.jpg")
    panel.save(panel_path, quality=95)

    metadata = {
        "dataset_index": int(index),
        "reference_key": sample.get("reference_key"),
        "expression": sample.get("expression"),
        "image_path": os.path.abspath(sample["image_path"]),
        "sigma": float(sigma),
        "artifacts": {
            "image": os.path.abspath(os.path.join(directory, f"{stem}_image.jpg")),
            "gt": os.path.abspath(os.path.join(directory, f"{stem}_gt.png")),
            "head_clean": os.path.abspath(os.path.join(directory, f"{stem}_head_clean.png")),
            "mask_clean": os.path.abspath(os.path.join(directory, f"{stem}_mask_clean.png")),
            "head_noisy": os.path.abspath(os.path.join(directory, f"{stem}_head_noisy.png")),
            "mask_noisy": os.path.abspath(os.path.join(directory, f"{stem}_mask_noisy.png")),
            "head_panel": os.path.abspath(panel_path),
        },
        "metrics": metrics,
    }
    with open(os.path.join(directory, f"{stem}.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def _all_reduce(value: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value


def save_curve_plot(path: str, curve: Iterable[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"Could not save curve plot because matplotlib is unavailable: {error}")
        return
    rows = list(curve)
    sigmas = [row["sigma"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    for key, label, style in (
        ("sam_cIoU", "SAM cIoU", "o-"),
        ("sam_gIoU", "SAM gIoU", "o--"),
        ("head_orig_cIoU", "Head cIoU", "s-"),
        ("head_orig_gIoU", "Head gIoU", "s--"),
    ):
        axis.plot(sigmas, [row[key] for row in rows], style, label=label)
    axis.set_xlabel("Relative latent noise sigma")
    axis.set_ylabel("IoU (%)")
    axis.set_title("llava6-3-8 Stage-I reconstruction robustness")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((TeacherEvalArguments, DataArguments))
    eval_args, data_args = parser.parse_args_into_dataclasses()
    rank, world, device = initialize_distributed()
    if not os.path.isfile(eval_args.teacher_checkpoint):
        raise FileNotFoundError(f"Missing Teacher checkpoint: {eval_args.teacher_checkpoint}")
    if int(eval_args.noise_repeats) <= 0:
        raise ValueError("noise_repeats must be positive.")
    sigmas = parse_floats(eval_args.noise_sigmas)
    visual_sigmas = parse_floats(eval_args.visual_sigmas) if eval_args.visual_sigmas.strip() else []
    for sigma in visual_sigmas:
        if sigma not in sigmas:
            sigmas.append(sigma)

    torch.manual_seed(int(eval_args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(eval_args.seed) + rank)

    teacher = MaskLatentTeacher(eval_args.sam_path, eval_args.sam_model_type)
    teacher.load_teacher(eval_args.teacher_checkpoint)
    teacher.to(device=device).eval()
    # Keep the learned latent and head in fp32; SAM towers use their checkpoint dtype.
    teacher.mask_encoder.float()
    teacher.mask_head.float()
    teacher.prompt_adapter.float()

    eval_data_args = type(data_args)(**{
        **data_args.__dict__,
        "train_split": eval_args.split,
        "train_subset_ratio": 1.0,
    })
    dataset = OriginalMaskReferDataset(eval_data_args)
    collator = TeacherCollator()
    limit = len(dataset) if int(eval_args.max_samples) < 0 else min(len(dataset), int(eval_args.max_samples))

    # [sam_inter, sam_union, sam_giou, sam_bce, sam_dice,
    #  head_orig_inter, head_orig_union, head_orig_giou, head_orig_bce,
    #  head_orig_dice, head_256_inter, head_256_union, head_256_giou,
    #  head_256_bce, head_256_dice, latent_error, area_fraction, count]
    sums = torch.zeros(len(sigmas), 18, device=device, dtype=torch.float64)
    visual_count = 0

    for index in range(rank, limit, world):
        sample = dataset[index]
        if int(sample["target_groups"]) == GROUP_EMPTY and not eval_args.include_empty:
            continue
        batch = collator([sample])
        sizes = batch["sam_input_sizes"].to(device)
        originals = batch["original_sizes"].to(device)
        latent, mask256 = teacher.encode_gt(batch["gt_masks"], sizes)
        sam_features = teacher.encode_sam_images(batch["sam_images"])
        target_orig = batch["gt_masks"][0].to(device=device, dtype=torch.bool).unsqueeze(0)
        target256 = mask256.to(device=device, dtype=torch.float32) >= 0.5
        fraction = float(target_orig.float().mean().item())
        head_clean_256 = teacher.decode_mask_head(latent)
        head_clean_orig = teacher.original_logits(head_clean_256, sizes, originals)[0]
        mask_clean_256 = teacher.decode_latent(latent, sam_features)
        mask_clean_orig = teacher.original_logits(mask_clean_256, sizes, originals)[0]
        clean_head_orig_pred = (head_clean_orig.sigmoid() >= float(eval_args.threshold)).squeeze().cpu().numpy()
        clean_mask_orig_pred = (mask_clean_orig.sigmoid() >= float(eval_args.threshold)).squeeze().cpu().numpy()

        for sigma_position, sigma in enumerate(sigmas):
            repeats = max(1, int(eval_args.noise_repeats))
            for repeat in range(repeats):
                if sigma == 0.0:
                    noisy = latent
                else:
                    noisy = perturb_latent(latent, sigma)
                head_noisy_256 = teacher.decode_mask_head(noisy)
                head_noisy_orig = teacher.original_logits(head_noisy_256, sizes, originals)[0]
                sam_256 = teacher.decode_latent(noisy, sam_features)
                sam_orig = teacher.original_logits(sam_256, sizes, originals)[0]

                sam_values = mask_metrics(sam_orig, target_orig, eval_args.threshold)
                head_orig_values = mask_metrics(head_noisy_orig, target_orig, eval_args.threshold)
                head_256_values = mask_metrics(head_noisy_256, target256, eval_args.threshold)
                rel_error = relative_latent_error(noisy, latent)
                values = (
                    sam_values[0], sam_values[1], sam_values[2], sam_values[3], sam_values[4],
                    head_orig_values[0], head_orig_values[1], head_orig_values[2], head_orig_values[3], head_orig_values[4],
                    head_256_values[0], head_256_values[1], head_256_values[2], head_256_values[3], head_256_values[4],
                    rel_error, rel_error.new_tensor(fraction), rel_error.new_tensor(1.0),
                )
                sums[sigma_position] += torch.stack([item.to(dtype=torch.float64) for item in values])

                if (
                    rank == 0
                    and bool(eval_args.save_visuals)
                    and sigma in visual_sigmas
                    and repeat == 0
                    and visual_count < int(eval_args.visual_max_samples)
                ):
                    head_noisy_pred = (head_noisy_orig.sigmoid() >= float(eval_args.threshold)).squeeze().cpu().numpy()
                    mask_noisy_pred = (sam_orig.sigmoid() >= float(eval_args.threshold)).squeeze().cpu().numpy()
                    gt_numpy = target_orig.squeeze().cpu().numpy()
                    visual_metrics = {
                        "head_clean_orig_gIoU": float(mask_metrics(head_clean_orig, target_orig, eval_args.threshold)[2].item() * 100.0),
                        "mask_clean_orig_gIoU": float(mask_metrics(mask_clean_orig, target_orig, eval_args.threshold)[2].item() * 100.0),
                        "head_noisy_orig_gIoU": float(head_orig_values[2].item() * 100.0),
                        "mask_noisy_orig_gIoU": float(sam_values[2].item() * 100.0),
                        "latent_relative_error": float(rel_error.item()),
                    }
                    save_head_visual(
                        eval_args.output_dir, index, rank, sigma, sample,
                        gt_numpy, clean_head_orig_pred, clean_mask_orig_pred,
                        head_noisy_pred, mask_noisy_pred, visual_metrics,
                    )
                    visual_count += 1

    _all_reduce(sums)
    if rank != 0:
        return

    os.makedirs(eval_args.output_dir, exist_ok=True)
    curve = []
    for position, sigma in enumerate(sigmas):
        row = sums[position]
        count = row[17].clamp_min(1.0)
        curve.append({
            "sigma": float(sigma),
            "actual_latent_relative_error": float((row[15] / count).item()),
            "sam_cIoU": float((100.0 * row[0] / row[1].clamp_min(1.0)).item()),
            "sam_gIoU": float((100.0 * row[2] / count).item()),
            "sam_BCE": float((row[3] / count).item()),
            "sam_DiceLoss": float((row[4] / count).item()),
            "head_orig_cIoU": float((100.0 * row[5] / row[6].clamp_min(1.0)).item()),
            "head_orig_gIoU": float((100.0 * row[7] / count).item()),
            "head_orig_BCE": float((row[8] / count).item()),
            "head_orig_DiceLoss": float((row[9] / count).item()),
            "head_256_cIoU": float((100.0 * row[10] / row[11].clamp_min(1.0)).item()),
            "head_256_gIoU": float((100.0 * row[12] / count).item()),
            "head_256_BCE": float((row[13] / count).item()),
            "head_256_DiceLoss": float((row[14] / count).item()),
            "mean_target_area_fraction": float((row[16] / count).item()),
            "count": int(row[17].item()),
        })

    metrics = {
        "format": "llava6_3_8_stage1_teacher_eval_v1",
        "teacher_checkpoint": os.path.abspath(eval_args.teacher_checkpoint),
        "split": eval_args.split,
        "threshold": float(eval_args.threshold),
        "max_samples": int(eval_args.max_samples),
        "noise_repeats": int(eval_args.noise_repeats),
        "include_empty": bool(eval_args.include_empty),
        "noise_curve": curve,
        "interpretation": {
            "sam_cIoU": "aggregate intersection / aggregate union on original image resolution",
            "sam_gIoU": "mean per-expression IoU on original image resolution",
            "head_orig": "image-free mask head postprocessed to original image geometry",
            "head_256": "image-free mask head on the 256x256 latent canvas",
        },
    }
    json_path = os.path.join(eval_args.output_dir, f"teacher_{eval_args.split}_noise_curve.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    csv_path = os.path.join(eval_args.output_dir, f"teacher_{eval_args.split}_noise_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    save_curve_plot(os.path.join(eval_args.output_dir, f"teacher_{eval_args.split}_noise_curve.png"), curve)
    print(json.dumps(metrics, indent=2))
    print(f"Saved Stage-I metrics: {json_path}")
    print(f"Saved Stage-I CSV: {csv_path}")
    if eval_args.save_visuals:
        print(f"Saved Stage-I head visualizations under: {os.path.join(eval_args.output_dir, 'visuals')}")


if __name__ == "__main__":
    main()





