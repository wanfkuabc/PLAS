"""llava6-3-8 evaluation: positive-only RefCOCO / RefCOCO+ / RefCOCOg.

This branch has no no-target head, so the report drops the probability sweep and
the prior correction and spends the space on what still moves the number:

* segmentation quality by object size (area fraction) crossed with the pixel
  threshold, so a size-specific weakness cannot hide inside an average;
* the IoU distribution, which separates "picked the wrong object" from
  "right object, loose boundary" -- the two need opposite fixes;
* how well the relative latent error predicts the outcome, which is the one
  quantity every round of this project has come back to.

A split that does contain no-target expressions still scores correctly: an
all-empty prediction counts as a rejection (the llava6-3-1 area rule).
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image

from PLAS.model.mask_latent_teacher import relative_latent_error
from PLAS.train.qwen_refseg_data import GROUP_EMPTY, GROUP_MULTIPLE, GROUP_SINGLE
from PLAS.train.train_qwen_dense_latent_refseg import (
    DataArguments, ModelArguments, QwenCollator, QwenReferDataset, build_student,
)


# COCO's 96^2 "large" cut is useless on RefCOCO: every referred object lands
# above it.  Bucket by area fraction of the image, which resolves structure
# inside that mass.  0.5% of a 640x480 image is roughly 39x39 px.
SIZE_EDGES = (0.005, 0.02, 0.05, 0.15, 0.35)
SIZE_NAMES = ("<0.5%", "0.5-2%", "2-5%", "5-15%", "15-35%", ">35%")
IOU_EDGES = (0.10, 0.50, 0.70, 0.85)
IOU_NAMES = ("wrong object <0.10", "partial 0.10-0.50", "loose 0.50-0.70",
             "good 0.70-0.85", "tight >0.85")


@dataclass
class EvalArguments:
    output_dir: str = "./PLAS/outputs/evaluation"
    split: str = "val"
    threshold: float = 0.55
    mask_thresholds: str = "0.40,0.45,0.50,0.55,0.60"
    empty_area_threshold: float = 0.0
    save_visuals: bool = True
    visual_max_samples: int = 256
    local_rank: int = -1
    # "sam" decodes the latent through the dense-prompt + SAM mask decoder;
    # "head" decodes the same latent through the image-free Stage-I head.
    # Toggling this isolates the decoder while holding the latent fixed.
    decode_path: str = "sam"
    # Inference-time quantization probe.  0 disables it; otherwise every latent
    # element is snapped to `latent_levels` evenly spaced values before decoding.
    # A flat curve down to a low level count means the decoder consumes a graded
    # value; a cliff localizes the effective resolution of the representation.
    latent_levels: int = 0
    # E12: extract frozen spatial features during the ordinary Stage-II-B
    # evaluation.  -1 processes the whole split; a positive value samples a
    # deterministic subset shared by all distributed ranks.
    save_probe_features: bool = False
    save_probe_layers: bool = False
    probe_max_samples: int = -1
    probe_seed: int = 42


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


def _mask_bbox(mask):
    """Return an original-image bbox as [x_min, y_min, x_max, y_max).

    The upper bounds are exclusive, matching normal Python image slicing.  A
    null bbox is used for an empty prediction/target instead of inventing a
    location.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_visual(output_dir, index, rank, sample, group, prediction, target,
                iou, latent_error, size_fraction, rejected):
    directory = os.path.join(output_dir, "visuals", f"rank_{rank}")
    os.makedirs(directory, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample["reference_key"])[:96]
    stem = f"{index:06d}_{key}"
    with Image.open(sample["image_path"]) as source:
        image = source.convert("RGB")
    predicted = np.squeeze(prediction.detach().cpu().numpy()).astype(bool)
    ground_truth = np.squeeze(target.detach().cpu().numpy()).astype(bool)
    pred_mask_path = os.path.join(directory, f"{stem}_pred.png")
    gt_mask_path = os.path.join(directory, f"{stem}_gt.png")
    image_path = os.path.join(directory, f"{stem}_image.jpg")
    pred_overlay_path = os.path.join(directory, f"{stem}_pred_overlay.jpg")
    overlay_path = os.path.join(directory, f"{stem}_overlay.jpg")
    json_path = os.path.join(directory, f"{stem}.json")

    Image.fromarray((predicted * 255).astype(np.uint8)).save(pred_mask_path)
    Image.fromarray((ground_truth * 255).astype(np.uint8)).save(gt_mask_path)
    image.save(image_path, quality=95)

    # Prediction-only visualization on the original-resolution RGB image.
    # Red is reserved for the predicted foreground; the GT/pred comparison
    # remains available separately as ``*_overlay.jpg`` below.
    pred_overlay = np.asarray(image).astype(np.float32)
    pred_overlay[predicted] = (
        0.55 * pred_overlay[predicted]
        + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    )
    Image.fromarray(pred_overlay.clip(0, 255).astype(np.uint8)).save(
        pred_overlay_path, quality=95
    )

    overlay = np.asarray(image).astype(np.float32)
    overlay[ground_truth] = 0.55 * overlay[ground_truth] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
    overlay[predicted] = 0.55 * overlay[predicted] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(overlay_path, quality=95)

    # Keep one metadata file per image/expression.  In addition to the prompt
    # and source image path, record the mask locations in original-image
    # coordinates so the saved artifact can be traced without reopening the
    # dataset annotations.
    source_image_path = os.path.abspath(sample["image_path"])
    metadata = {
        "reference_key": sample["reference_key"],
        "text_prompt": sample["expression"],
        "expression": sample["expression"],
        "image": {
            "path": source_image_path,
            "file_name": os.path.basename(source_image_path),
            "width": int(image.width),
            "height": int(image.height),
        },
        "image_path": source_image_path,
        "image_position": {
            "coordinate_system": "original_image_pixels",
            "pred_bbox_xyxy": _mask_bbox(predicted),
            "gt_bbox_xyxy": _mask_bbox(ground_truth),
            "bbox_format": "[x_min, y_min, x_max, y_max), upper bounds exclusive",
        },
        "artifacts": {
            "original_image": os.path.abspath(image_path),
            "pred_mask": os.path.abspath(pred_mask_path),
            "gt_mask": os.path.abspath(gt_mask_path),
            "pred_overlay": os.path.abspath(pred_overlay_path),
            "gt_pred_overlay": os.path.abspath(overlay_path),
        },
        "target_group": group,
        "overlay": "pred_overlay: red=prediction; gt_pred_overlay: green=GT, red=prediction",
        "iou": float(iou),
        "latent_relative_error": float(latent_error),
        "target_area_fraction": float(size_fraction),
        "rejected": bool(rejected),
        "failure_type": (
            "rejected_or_empty" if rejected else
            "wrong_object" if iou < 0.10 else
            "partial" if iou < 0.50 else
            "loose_boundary" if iou < 0.70 else
            "good" if iou < 0.85 else "tight"
        ),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def _bucket(value, edges):
    for position, edge in enumerate(edges):
        if value < edge:
            return position
    return len(edges)


def _correlation(first, second):
    if len(first) < 3:
        return float("nan")
    mean_a, mean_b = statistics.fmean(first), statistics.fmean(second)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second))
    spread_a = sum((a - mean_a) ** 2 for a in first) ** 0.5
    spread_b = sum((b - mean_b) ** 2 for b in second) ** 0.5
    return numerator / (spread_a * spread_b) if spread_a > 0 and spread_b > 0 else float("nan")


def save_probe_sample(output_dir, split, index, rank, sample, record, probe,
                      mask256, prediction, latent, sam_input_size,
                      save_layers=False, grid_size=16):
    """Persist one independent sample; the model and its checkpoint stay frozen."""
    directory = os.path.join(output_dir, "e12_readout_probe", split, f"rank_{rank}")
    os.makedirs(directory, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample["reference_key"]))[:80]
    stem = f"{index:06d}_{key}"
    gt_path = os.path.join(directory, f"{stem}_gt.png")
    pred_path = os.path.join(directory, f"{stem}_pred.png")
    gt_original = np.asarray(sample["gt_masks"].squeeze(0).cpu(), dtype=np.uint8)
    pred_original = np.asarray(prediction.squeeze().detach().cpu(), dtype=np.uint8)
    Image.fromarray(gt_original * 255).save(gt_path)
    Image.fromarray(pred_original * 255).save(pred_path)
    # Teacher's mask256() applies the same top-left SAM resize/pad transform as
    # the Qwen image canvas.  Area downsampling gives one occupancy per cell.
    occupancy = F.interpolate(mask256.float(), size=(grid_size, grid_size), mode="area")[0, 0]
    features = {
        name: values[0].detach().to(device="cpu", dtype=torch.float16).contiguous()
        for name, values in probe.items() if name != "readout_layers"
    }
    layer_features = ({
        name: values[0].detach().to(device="cpu", dtype=torch.float16).contiguous()
        for name, values in probe["readout_layers"].items()
    } if save_layers else {})
    payload = {
        "index": int(index), "reference_key": str(sample["reference_key"]),
        "image_id": int(record[5]), "annotation_ids": list(record[6]),
        "image_path": os.path.abspath(sample["image_path"]),
        "expression": str(sample["expression"]),
        "sam_input_size": [int(value) for value in sam_input_size],
        "gt_mask_path": os.path.abspath(gt_path), "pred_mask_path": os.path.abspath(pred_path),
        "target_occupancy": occupancy.detach().cpu().to(dtype=torch.float16),
        "target_grid": (occupancy >= 0.5).detach().cpu().to(dtype=torch.uint8),
        "features": features, "readout_layers": layer_features,
        "student_latent": latent[0].detach().to(device="cpu", dtype=torch.float16).contiguous(),
        "feature_semantics": {
            "readout": "grid projection of fused Qwen readout states; before visual fusion",
            "visual": "vision projection of Qwen image states; before readout fusion",
            "joint": "after visual fusion and text-spatial attention; before latent adapter",
        },
    }
    torch.save(payload, os.path.join(directory, f"{stem}.pt"))


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    rank, world, device = initialize_distributed()
    if not os.path.isfile(model_args.teacher_checkpoint) or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("Evaluation requires --teacher_checkpoint and --student_checkpoint.")
    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()

    eval_data_args = DataArguments(**{
        **data_args.__dict__, "train_split": eval_args.split, "train_subset_ratio": 1.0,
        "synthetic_negative_ratio": 0.0,
    })
    dataset = QwenReferDataset(eval_data_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)
    indices = list(range(len(dataset)))
    if eval_args.probe_max_samples > 0 and eval_args.probe_max_samples < len(indices):
        indices = sorted(random.Random(eval_args.probe_seed).sample(indices, eval_args.probe_max_samples))

    mask_values = sorted(set(
        [float(item) for item in str(eval_args.mask_thresholds).split(",") if item.strip()]
        + [float(eval_args.threshold)]
    ))
    headline = mask_values.index(float(eval_args.threshold))

    # [gIoU_sum, count, cIoU_i, cIoU_u, empty_ok, empty_n, pos_iou, pos_n]
    totals = torch.zeros(8, device=device, dtype=torch.float64)
    size_sweep = torch.zeros(len(mask_values), len(SIZE_NAMES), 2, device=device, dtype=torch.float64)
    iou_bands = torch.zeros(len(IOU_NAMES), device=device, dtype=torch.float64)
    detail = torch.zeros(8, device=device, dtype=torch.float64)
    samples = []

    for position in range(rank, len(indices), world):
        index = indices[position]
        sample = dataset[index]
        batch = collator([sample])
        qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
        sam_input_sizes = batch["sam_input_sizes"].to(device)
        original_sizes = batch["original_sizes"].to(device)
        result = model.predict(
            qwen_inputs, batch["sam_images"], sam_input_sizes, original_sizes,
            decode_path=eval_args.decode_path,
            latent_levels=eval_args.latent_levels,
            return_probe_features=eval_args.save_probe_features,
        )
        logits, latent = result[:2]
        logit = logits[0]
        probability_map = logit.sigmoid()
        target = batch["gt_masks"][0].to(device=device, dtype=torch.bool)
        group = int(batch["target_groups"].item())

        target_latent, mask256 = model.teacher.encode_gt(batch["gt_masks"], sam_input_sizes)
        error = float(relative_latent_error(latent, target_latent.to(latent.device)))

        prediction = probability_map >= float(eval_args.threshold)
        intersection = (prediction & target).sum(dtype=torch.float64)
        union = (prediction | target).sum(dtype=torch.float64)
        target_area = target.sum(dtype=torch.float64)
        rejected = float(prediction.float().mean(dtype=torch.float64)) <= float(eval_args.empty_area_threshold)
        sample_iou = 0.0 if rejected else float(intersection / union.clamp_min(1.0))
        size_fraction = float(target.float().mean(dtype=torch.float64))

        if eval_args.save_probe_features:
            save_probe_sample(eval_args.output_dir, eval_args.split, index, rank, sample,
                              dataset.records[index], result[2], mask256,
                              prediction, latent, sam_input_sizes[0],
                              save_layers=eval_args.save_probe_layers)

        if group == GROUP_EMPTY:
            totals[1] += 1; totals[5] += 1
            totals[0] += float(rejected); totals[4] += float(rejected)
            if not rejected:
                totals[3] += union
        else:
            iou = sample_iou
            totals[0] += iou; totals[1] += 1; totals[6] += iou; totals[7] += 1
            totals[2] += 0.0 if rejected else intersection
            totals[3] += target_area if rejected else union
            iou_bands[_bucket(iou, IOU_EDGES)] += 1
            offset = 0 if group == GROUP_SINGLE else 4
            detail[offset + 0] += iou
            detail[offset + 1] += 1
            detail[offset + 2] += 0.0 if rejected else intersection
            detail[offset + 3] += target_area if rejected else union

            size_bucket = _bucket(size_fraction, SIZE_EDGES)
            for position, mask_threshold in enumerate(mask_values):
                candidate = probability_map >= mask_threshold
                overlap = (candidate & target).sum(dtype=torch.float64)
                span = (candidate | target).sum(dtype=torch.float64).clamp_min(1.0)
                size_sweep[position, size_bucket, 0] += overlap / span
                size_sweep[position, size_bucket, 1] += 1.0
            samples.append((error, iou))

        if eval_args.save_visuals and (eval_args.visual_max_samples < 0 or index < eval_args.visual_max_samples):
            save_visual(eval_args.output_dir, index, rank, sample, group,
                        torch.zeros_like(prediction) if rejected else prediction, target,
                        sample_iou, error, size_fraction, rejected)

    if torch.distributed.is_initialized():
        for tensor in (totals, size_sweep, iou_bands, detail):
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        gathered = [None for _ in range(world)]
        torch.distributed.all_gather_object(gathered, samples)
        samples = [item for shard in gathered for item in shard]

    if rank != 0:
        return

    errors = [error for error, _ in samples]
    ious = [iou for _, iou in samples]
    positives = int(totals[7].item())
    metrics = {
        "split": eval_args.split,
        "evaluated_samples": len(indices),
        "threshold": float(eval_args.threshold),
        "decode_path": eval_args.decode_path,
        "latent_levels": int(eval_args.latent_levels),
        "gIoU": float((100.0 * totals[0] / totals[1].clamp_min(1)).item()),
        "cIoU": float((100.0 * totals[2] / totals[3].clamp_min(1)).item()),
        "positive_gIoU": float((100.0 * totals[6] / totals[7].clamp_min(1)).item()),
        "N-acc": float((100.0 * totals[4] / totals[5].clamp_min(1)).item()),
        "counts": {
            "positive": positives,
            "single": int(detail[1].item()),
            "multiple": int(detail[5].item()),
            "no_target": int(totals[5].item()),
        },
        "by_target_group": {
            "single_gIoU": float((100.0 * detail[0] / detail[1].clamp_min(1)).item()),
            "single_cIoU": float((100.0 * detail[2] / detail[3].clamp_min(1)).item()),
            "multiple_gIoU": float((100.0 * detail[4] / detail[5].clamp_min(1)).item()),
            "multiple_cIoU": float((100.0 * detail[6] / detail[7].clamp_min(1)).item()),
        },
        "iou_distribution": {
            IOU_NAMES[position]: {
                "count": int(iou_bands[position].item()),
                "share_percent": float(100.0 * iou_bands[position].item() / max(1, positives)),
            }
            for position in range(len(IOU_NAMES))
        },
        "positive_quality_by_object_size": {
            "counts": {
                SIZE_NAMES[b]: int(size_sweep[headline, b, 1].item()) for b in range(len(SIZE_NAMES))
            },
            "gIoU": [
                {
                    "mask_threshold": float(mask_values[p]),
                    **{
                        SIZE_NAMES[b]: float(
                            (100.0 * size_sweep[p, b, 0] / size_sweep[p, b, 1].clamp_min(1)).item()
                        )
                        for b in range(len(SIZE_NAMES))
                    },
                }
                for p in range(len(mask_values))
            ],
        },
        "latent": {
            "mean_relative_error": statistics.fmean(errors) if errors else float("nan"),
            "median_relative_error": statistics.median(errors) if errors else float("nan"),
            "p90_relative_error": float(np.percentile(errors, 90)) if errors else float("nan"),
            "pearson_error_vs_iou": _correlation(errors, ious),
            # Frozen-decoder reference measured on llava6-3-2 (grefcoco testA):
            # IoU 80.57 at error 0.191, slope -125.2 IoU per unit error.
            "alpha_curve_reference_iou": (
                80.57 - 125.2 * (statistics.fmean(errors) - 0.191) if errors else float("nan")
            ),
        },
        "learned_gates": model.calibration_scales(),
    }
    if errors:
        metrics["latent"]["decoder_adaptation_gain"] = (
            metrics["positive_gIoU"] - metrics["latent"]["alpha_curve_reference_iou"]
        )
    os.makedirs(eval_args.output_dir, exist_ok=True)
    # Keep each decoder path and quantization level side by side instead of
    # letting a sweep overwrite its own earlier points.
    level_tag = f"_q{int(eval_args.latent_levels)}" if int(eval_args.latent_levels) > 0 else ""
    path = os.path.join(
        eval_args.output_dir,
        f"{eval_args.split}_{eval_args.decode_path}{level_tag}_metrics.json",
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved llava6-3-8 evaluation: {path}")


if __name__ == "__main__":
    main()





