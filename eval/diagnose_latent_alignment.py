"""The five llava6-3-4 diagnostics, in one pass over the validation split.

1. Latent interpolation curve.  Z(alpha) = normalize(Z_gt + alpha * (Z_pred -
   Z_gt)) decoded through the real path.  A sharply concave curve means the
   latent space is still not smooth; a near-linear curve means the Student's
   prediction error is simply too large and the Qwen side is the bottleneck.
2. Per-sample relative latent error against IoU, with Pearson and Spearman
   correlations.  A weak correlation falsifies latent alignment as a proxy.
3. Error attribution: "wrong object" versus "right object, bad boundary".
4. The learned calibrator scales and readout-layer mixture.
5. The no-target head's accuracy, separately from pixel metrics.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import transformers

from PLAS.model.mask_latent_teacher import normalize_latent, relative_latent_error
from PLAS.train.qwen_refseg_data import GROUP_EMPTY
from PLAS.train.train_qwen_dense_latent_refseg import (
    DataArguments, ModelArguments, QwenCollator, QwenReferDataset, build_student,
)


@dataclass
class DiagnoseArguments:
    output_dir: str = "./PLAS/outputs/diagnostics"
    split: str = "val"
    threshold: float = 0.55
    max_samples: int = 1000
    alphas: str = "0.0,0.25,0.5,0.75,1.0"
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


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _centroid(mask: torch.Tensor):
    positions = torch.nonzero(mask, as_tuple=False)
    if positions.numel() == 0:
        return None
    return positions.float().mean(dim=0)


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DiagnoseArguments))
    model_args, data_args, diagnose_args = parser.parse_args_into_dataclasses()
    rank, world, device = initialize_distributed()
    if not os.path.isfile(model_args.teacher_checkpoint) or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("Diagnostics require --teacher_checkpoint and --student_checkpoint.")
    alphas = [float(item) for item in str(diagnose_args.alphas).split(",") if item.strip()]

    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()

    eval_data_args = DataArguments(**{**data_args.__dict__, "train_split": diagnose_args.split, "train_subset_ratio": 1.0,
        # Synthetic negatives are a *training* device.  Never let them
        # into an evaluation split: they would inflate N-acc and distort
        # the split prior used for the threshold correction.
        "synthetic_negative_ratio": 0.0,
    })
    dataset = QwenReferDataset(eval_data_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)
    limit = len(dataset) if int(diagnose_args.max_samples) < 0 else min(len(dataset), int(diagnose_args.max_samples))

    alpha_sums = torch.zeros(len(alphas), device=device, dtype=torch.float64)
    alpha_count = torch.zeros((), device=device, dtype=torch.float64)
    rows = []

    for index in range(rank, limit, world):
        sample = dataset[index]
        batch = collator([sample])
        qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
        sizes = batch["sam_input_sizes"].to(device)
        originals = batch["original_sizes"].to(device)
        group = int(batch["target_groups"].item())
        target = batch["gt_masks"][0].to(device=device, dtype=torch.bool)

        latent = model.encode(qwen_inputs)
        target_latent, _ = model.teacher.encode_gt(batch["gt_masks"], sizes)
        target_latent = target_latent.to(latent.device)
        error = float(relative_latent_error(latent, target_latent))
        if group == GROUP_EMPTY:
            continue

        sam_features = model.teacher.encode_sam_images(batch["sam_images"])
        ious = []
        for position, alpha in enumerate(alphas):
            mixed = normalize_latent(target_latent + float(alpha) * (latent - target_latent))
            masks256 = model.decode_with_sam_features(mixed, sam_features)
            logit = model.original_logits_from_sizes(masks256, sizes, originals)[0]
            prediction = logit.sigmoid() >= float(diagnose_args.threshold)
            intersection = (prediction & target).sum(dtype=torch.float64)
            union = (prediction | target).sum(dtype=torch.float64).clamp_min(1.0)
            iou = float(intersection / union)
            ious.append(iou)
            alpha_sums[position] += iou
            if position == len(alphas) - 1:
                centroid = _centroid(prediction[0, 0] if prediction.ndim == 4 else prediction.squeeze())
                inside = False
                if centroid is not None:
                    flat_target = target[0] if target.ndim == 3 else target
                    row = int(centroid[0].clamp(0, flat_target.shape[0] - 1))
                    column = int(centroid[1].clamp(0, flat_target.shape[1] - 1))
                    inside = bool(flat_target[row, column])
                rows.append({
                    "index": index,
                    "reference_key": sample["reference_key"],
                    "expression": sample["expression"],
                    "target_group": group,
                    "relative_latent_error": error,
                    "iou": iou,
                    "centroid_inside_gt": int(inside),
                    "predicted_empty": int(prediction.any().item() == 0),
                })
        alpha_count += 1.0

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(alpha_sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(alpha_count, op=torch.distributed.ReduceOp.SUM)
        gathered = [None for _ in range(world)]
        torch.distributed.all_gather_object(gathered, rows)
        rows = [row for shard in gathered for row in shard]

    if rank != 0:
        return

    os.makedirs(diagnose_args.output_dir, exist_ok=True)
    csv_path = os.path.join(diagnose_args.output_dir, f"{diagnose_args.split}_per_sample.csv")
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    errors = np.array([row["relative_latent_error"] for row in rows], dtype=np.float64)
    ious = np.array([row["iou"] for row in rows], dtype=np.float64)
    wrong_indices = {
        row["index"] for row in rows if row["iou"] < 0.10 or not row["centroid_inside_gt"]
    }
    wrong_object = [row for row in rows if row["index"] in wrong_indices]
    coarse = [row for row in rows if row["index"] not in wrong_indices and row["iou"] < 0.70]

    report = {
        "split": diagnose_args.split,
        "samples": int(alpha_count.item()),
        "latent_interpolation_curve": [
            {"alpha": alpha, "gIoU": float(100.0 * alpha_sums[position].item() / max(1.0, alpha_count.item()))}
            for position, alpha in enumerate(alphas)
        ],
        "latent_error_vs_iou": {
            "pearson": _correlation(errors, ious) if len(rows) else float("nan"),
            "spearman": _correlation(_rank_values(errors), _rank_values(ious)) if len(rows) else float("nan"),
            "mean_relative_latent_error": float(errors.mean()) if len(rows) else float("nan"),
            "p90_relative_latent_error": float(np.percentile(errors, 90)) if len(rows) else float("nan"),
        },
        "error_attribution": {
            "positive_samples": len(rows),
            "wrong_object": len(wrong_object),
            "wrong_object_ratio": float(len(wrong_object) / max(1, len(rows))),
            "right_object_coarse_boundary": len(coarse),
            "right_object_coarse_ratio": float(len(coarse) / max(1, len(rows))),
        },
        "calibration": model.calibration_scales(),
        "per_sample_csv": csv_path,
    }
    path = os.path.join(diagnose_args.output_dir, f"{diagnose_args.split}_diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved llava6-3-4 diagnostics: {path}")


if __name__ == "__main__":
    main()





