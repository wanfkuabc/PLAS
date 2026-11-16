"""Tier-2 continuity probe: spherical interpolation between two latents.

The question this answers is not "is the output a float" but "does the decoder
respond to the latent as a graded coordinate, or as a lookup into a small set
of codes".  The design keeps everything else fixed:

    z_t = slerp(z_student, z_teacher; t),  t in {0, 1/(S-1), ..., 1}

* Same image, same expression, same SAM features for every t.  Only the latent
  moves.
* The two endpoints are meaningful: t=0 is what the student predicts, t=1 is
  the teacher latent encoded from the ground-truth mask, which the decoder
  reconstructs at ~98% gIoU.  So the curve must rise from the student's
  accuracy to near-perfect, and the interesting question is *how* it rises.
* Interpolation is spherical, not linear.  ``normalize_latent`` puts every
  position on a unit sphere, so a straight line between two points passes
  through the interior and leaves the manifold.  Slerp keeps every probe point
  on the same sphere the decoder was trained on, so a drop in accuracy cannot
  be blamed on off-manifold inputs.

Reading the result:

* A graded decoder produces a proportional response: the fraction of the
  accuracy gap closed at t should track t itself (~0.25 at t=0.25, ~0.5 at
  t=0.5).  ``gap_closure`` reports exactly this.
* A codebook-like decoder produces a step: the mask stays at the student's
  answer, then snaps to the teacher's answer at some threshold t.
  ``monotone_fraction`` stays high but ``gap_closure`` is wildly non-linear
  (near 0 for most t, then a jump).

Both are reported, so the figure cannot be over-read in either direction.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from PIL import Image

from PLAS.model.mask_latent_teacher import normalize_latent, relative_latent_error
from PLAS.train.qwen_refseg_data import GROUP_EMPTY
from PLAS.train.train_qwen_dense_latent_refseg import (
    DataArguments, ModelArguments, QwenCollator, QwenReferDataset, build_student,
)


@dataclass
class InterpArguments:
    output_dir: str = "./PLAS/outputs/latent_continuity"
    split: str = "val"
    threshold: float = 0.55
    # Number of interpolation points including both endpoints.
    steps: int = 5
    max_samples: int = -1
    save_visuals: bool = True
    visual_max_samples: int = 8
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


def slerp(first: torch.Tensor, second: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical interpolation between two tensors on a per-position sphere.

    Channel dimension is dim=1, matching ``normalize_latent``.  Positions whose
    endpoints are nearly parallel fall back to a linear blend, which is exact
    in that limit.  The result is re-normalized so numerical drift cannot push
    a probe point off the sphere.
    """
    a = first.float()
    b = second.float()
    cosine = (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(cosine)
    sine = torch.sin(omega)
    near = sine.abs() < eps
    safe_sine = torch.where(near, torch.ones_like(sine), sine)
    weight_a = torch.where(near, torch.full_like(omega, 1.0 - t), torch.sin((1.0 - t) * omega) / safe_sine)
    weight_b = torch.where(near, torch.full_like(omega, t), torch.sin(t * omega) / safe_sine)
    blended = weight_a * a + weight_b * b
    return normalize_latent(blended)


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def _save_strip(output_dir: str, index: int, sample: dict, masks: Sequence[np.ndarray],
                fraction: float, rank: int) -> None:
    """One horizontal strip: the input image, the GT, then every t."""
    directory = os.path.join(output_dir, "strips", f"rank_{rank}")
    os.makedirs(directory, exist_ok=True)
    with Image.open(sample["image_path"]) as source:
        image = source.convert("RGB").resize((192, 192))
    panels = [np.asarray(image).astype(np.uint8)]
    for mask in masks:
        panel = np.asarray(image).astype(np.float32)
        panel[mask.astype(bool)] = 0.45 * panel[mask.astype(bool)] + 0.55 * np.array([255, 0, 0], np.float32)
        panels.append(panel.astype(np.uint8))
    strip = np.concatenate(panels, axis=1)
    name = f"{index:06d}_gap{fraction:.2f}.png"
    Image.fromarray(strip).save(os.path.join(directory, name))


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 3:
        return float("nan")
    mean_a, mean_b = statistics.fmean(first), statistics.fmean(second)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second))
    spread_a = sum((a - mean_a) ** 2 for a in first) ** 0.5
    spread_b = sum((b - mean_b) ** 2 for b in second) ** 0.5
    return numerator / (spread_a * spread_b) if spread_a > 0 and spread_b > 0 else float("nan")


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InterpArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    rank, world, device = initialize_distributed()
    if not os.path.isfile(model_args.teacher_checkpoint) or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("Interpolation probe requires --teacher_checkpoint and --student_checkpoint.")
    if int(args.steps) < 2:
        raise ValueError("--steps must be at least 2 (the two endpoints).")

    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()

    eval_data_args = DataArguments(**{
        **data_args.__dict__, "train_split": args.split, "train_subset_ratio": 1.0,
        "synthetic_negative_ratio": 0.0,
    })
    dataset = QwenReferDataset(eval_data_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)

    steps = int(args.steps)
    ts = [position / (steps - 1) for position in range(steps)]
    # per-t accumulators: sum(I), sum(U) for cIoU, plus per-sample IoU for gIoU
    sam_inter = [0.0] * steps
    sam_union = [0.0] * steps
    head_inter = [0.0] * steps
    head_union = [0.0] * steps
    sam_ious: List[List[float]] = [[] for _ in range(steps)]
    head_ious: List[List[float]] = [[] for _ in range(steps)]
    endpoint_gaps: List[float] = []
    student_errors: List[float] = []

    limit = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))

    for index in range(rank, limit, world):
        sample = dataset[index]
        batch = collator([sample])
        if int(batch["target_groups"].item()) == GROUP_EMPTY:
            continue                       # interpolation is undefined without a target
        qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
        sam_input_sizes = batch["sam_input_sizes"].to(device)
        original_sizes = batch["original_sizes"].to(device)
        target = np.squeeze(batch["gt_masks"][0].cpu().numpy()).astype(bool)

        student_latent = model.encode(qwen_inputs)
        teacher_latent, _ = model.teacher.encode_gt(batch["gt_masks"], sam_input_sizes)
        teacher_latent = teacher_latent.to(device=student_latent.device, dtype=student_latent.dtype)
        student_errors.append(float(relative_latent_error(student_latent, teacher_latent)))

        sam_features = model.teacher.encode_sam_images(batch["sam_images"])
        masks_for_strip: List[np.ndarray] = []
        sam_curve: List[float] = []
        head_curve: List[float] = []

        for position, t in enumerate(ts):
            probe = student_latent if t == 0.0 else (
                teacher_latent if t == 1.0 else slerp(student_latent, teacher_latent, t))

            sam_masks = model.decode_with_sam_features(probe, sam_features)
            sam_logits = model.original_logits_from_sizes(sam_masks, sam_input_sizes, original_sizes)[0]
            sam_prediction = (sam_logits.sigmoid() >= float(args.threshold)).cpu().numpy().squeeze()
            head_masks = model.decode_with_head(probe)
            head_logits = model.original_logits_from_sizes(head_masks, sam_input_sizes, original_sizes)[0]
            head_prediction = (head_logits.sigmoid() >= float(args.threshold)).cpu().numpy().squeeze()

            for prediction, inter, union, curve in (
                (sam_prediction, sam_inter, sam_union, sam_curve),
                (head_prediction, head_inter, head_union, head_curve),
            ):
                curve.append(iou(prediction, target))
                inter[position] += float(np.logical_and(prediction.astype(bool), target).sum())
                union[position] += float(np.logical_or(prediction.astype(bool), target).sum())

            sam_ious[position].append(sam_curve[-1])
            head_ious[position].append(head_curve[-1])
            masks_for_strip.append(sam_prediction)

        gap = sam_curve[-1] - sam_curve[0]
        endpoint_gaps.append(gap)
        if (args.save_visuals and abs(gap) > 0.15
                and len(endpoint_gaps) <= int(args.visual_max_samples)):
            _save_strip(args.output_dir, index, sample, masks_for_strip, gap, rank)

    if torch.distributed.is_initialized():
        for values in (sam_inter, sam_union, head_inter, head_union):
            tensor = torch.tensor(values, device=device, dtype=torch.float64)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            values[:] = tensor.tolist()
        gathered = [None for _ in range(world)]
        torch.distributed.all_gather_object(
            gathered, (sam_ious, head_ious, endpoint_gaps, student_errors))
        sam_ious = [sum(part, []) for part in zip(*[g[0] for g in gathered])]
        head_ious = [sum(part, []) for part in zip(*[g[1] for g in gathered])]
        endpoint_gaps = sum([g[2] for g in gathered], [])
        student_errors = sum([g[3] for g in gathered], [])

    if rank != 0:
        return

    def curve_summary(inter, union, ious):
        ciou = [100.0 * i / u if u > 0 else 0.0 for i, u in zip(inter, union)]
        giou = [100.0 * statistics.fmean(values) if values else 0.0 for values in ious]
        return ciou, giou

    sam_ciou, sam_giou = curve_summary(sam_inter, sam_union, sam_ious)
    head_ciou, head_giou = curve_summary(head_inter, head_union, head_ious)

    # How much of the endpoint-to-endpoint gap is closed at each t.  A graded
    # decoder tracks t; a step-like decoder stays near 0 then jumps to 1.
    def gap_closure(giou):
        start, end = giou[0], giou[-1]
        span = end - start
        if abs(span) < 1e-6:
            return [float("nan")] * len(giou)
        return [(value - start) / span for value in giou]

    sam_closure = gap_closure(sam_giou)
    head_closure = gap_closure(head_giou)

    def monotone_fraction(ious):
        """Share of samples whose IoU never decreases as t grows."""
        if not ious or not ious[0]:
            return float("nan")
        count = 0
        for sample in range(len(ious[0])):
            values = [curve[sample] for curve in ious]
            if all(later >= earlier - 1e-9 for earlier, later in zip(values, values[1:])):
                count += 1
        return count / len(ious[0])

    metrics = {
        "split": args.split,
        "steps": steps,
        "t": ts,
        "samples": len(sam_ious[0]) if sam_ious else 0,
        "student_relative_latent_error_mean": float(statistics.fmean(student_errors)) if student_errors else float("nan"),
        "sam": {
            "cIoU": sam_ciou, "gIoU": sam_giou, "gap_closure": sam_closure,
            "monotone_fraction": monotone_fraction(sam_ious),
            "correlation_with_t": _correlation(ts, sam_giou),
        },
        "head": {
            "cIoU": head_ciou, "gIoU": head_giou, "gap_closure": head_closure,
            "monotone_fraction": monotone_fraction(head_ious),
            "correlation_with_t": _correlation(ts, head_giou),
        },
        "note": (
            "t=0 is the student latent, t=1 the teacher latent from the ground-truth mask. "
            "A decoder that treats the latent as a graded coordinate closes the accuracy gap "
            "roughly in proportion to t; a codebook-like decoder stays flat and then jumps."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.split}_interpolation_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\n{'t':>6} {'SAM cIoU':>10} {'SAM gIoU':>10} {'closed':>8} | {'head cIoU':>10} {'head gIoU':>10}")
    for position, t in enumerate(ts):
        print(f"{t:6.2f} {sam_ciou[position]:10.2f} {sam_giou[position]:10.2f} "
              f"{sam_closure[position]:8.2f} | {head_ciou[position]:10.2f} {head_giou[position]:10.2f}")
    print(f"\nmonotone fraction  SAM={metrics['sam']['monotone_fraction']:.3f} "
          f"head={metrics['head']['monotone_fraction']:.3f}")
    print(f"Saved interpolation probe: {path}")


if __name__ == "__main__":
    main()





