"""Create publication figures for llava6-3-8 Stage-2 latent alignment.

The script compares the inference-time Student latent
``z_S = Adapter(Qwen3-VL(image, expression))`` with the Stage-1 Teacher target
``z_T = MaskEncoder(GT mask)``.  The GT mask is used only to construct the
offline diagnostic target and is never passed to the Student input.

Three figure pairs are written to ``--output_dir``:

* ``01_latent_alignment_pipeline``: training/inference flow diagram;
* ``02_latent_similarity_heatmap_*``: selected samples with GT/decoded masks
  and a per-position cosine-similarity heatmap;
* ``03_latent_similarity_quantitative``: dataset-level distributions and the
  relation between latent alignment and decoded-mask IoU;
* ``04_decode_path_comparison`` (``--decode_comparison True``): the same
  Student latent decoded by the image-free mask head and by the
  image-conditioned SAM path, scored at the original image resolution with
  dataset-level cIoU (accumulated intersection / accumulated union) and gIoU
  (mean per-sample IoU).

The per-sample ``head_ciou`` column of ``03`` is a per-sample IoU on the
256x256 canvas; its mean is a gIoU, not a cIoU.  Use the ``decode_comparison``
block of the JSON summary for numbers comparable to the main eval script.

The same script evaluates either ``latent_warmup`` (Stage 2A) or
``task_adaptation`` (Stage 2B); the phase must match the checkpoint and model
hyperparameters passed on the command line.

Each figure is saved as both PNG and PDF.  A CSV and JSON summary are also
written so the plotted numbers can be traced back to individual samples.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from matplotlib.patches import FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PLAS.train.train_qwen_dense_latent_refseg import (  # noqa: E402
    DataArguments,
    ModelArguments,
    QwenCollator,
    QwenReferDataset,
    build_student,
)


@dataclass
class FigureArguments:
    output_dir: str = "./PLAS/outputs/latent_alignment_figures"
    split: str = "val"
    max_samples: int = 0
    threshold: float = 0.55
    visual_index: int = 0
    visual_indices: str = ""
    visual_count: int = 256
    dpi: int = 220
    # Decode each Student latent through both the image-free head and the SAM
    # path, and score both at the original image resolution.
    decode_comparison: bool = True


def _save_figure(figure, stem: str, output_dir: Path, dpi: int) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=int(dpi), bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", metadata={"Creator": "llava6-3-8"})
    plt.close(figure)


def _box(ax, x: float, y: float, width: float, height: float, text: str, color: str):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.16,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2.0, y + height / 2.0, text,
            ha="center", va="center", fontsize=11, color="#1f2937",
            wrap=True)


def draw_pipeline(output_dir: Path, dpi: int, training_phase: str) -> None:
    figure, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    phase_label = "Stage 2A / latent warmup" if training_phase == "latent_warmup" else "Stage 2B / task adaptation"
    ax.text(0.5, 0.96, f"{phase_label} latent alignment", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#111827")
    ax.text(0.5, 0.90, "The GT-derived teacher latent is an offline training target; it is not an inference input.",
            ha="center", va="center", fontsize=10, color="#4b5563")

    # Student path.
    _box(ax, 0.03, 0.54, 0.14, 0.16, "Image +\nexpression", "#2563eb")
    _box(ax, 0.23, 0.54, 0.15, 0.16, "Qwen3-VL\n8B", "#2563eb")
    _box(ax, 0.44, 0.54, 0.16, 0.16, "Latent\nAdapter", "#2563eb")
    _box(ax, 0.66, 0.54, 0.13, 0.16, "Student latent\nz_S", "#2563eb")
    _box(ax, 0.84, 0.54, 0.13, 0.16, "Decoded\nstudent mask", "#16a34a")
    for start, end in [(0.17, 0.23), (0.38, 0.44), (0.60, 0.66), (0.79, 0.84)]:
        ax.annotate("", xy=(end, 0.62), xytext=(start, 0.62),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))

    # Teacher target path, intentionally dashed.
    _box(ax, 0.03, 0.16, 0.14, 0.16, "GT mask\n(training only)", "#d97706")
    _box(ax, 0.23, 0.16, 0.15, 0.16, "Stage-I Mask\nEncoder", "#d97706")
    _box(ax, 0.44, 0.16, 0.16, 0.16, "Teacher target\nlatent z_T", "#d97706")
    _box(ax, 0.84, 0.16, 0.13, 0.16, "Decoded\nteacher mask", "#16a34a")
    for start, end in [(0.17, 0.23), (0.38, 0.44), (0.60, 0.84)]:
        ax.annotate("", xy=(end, 0.24), xytext=(start, 0.24),
                    arrowprops=dict(arrowstyle="->", lw=1.7, color="#d97706", linestyle="--"))

    # Alignment and shared decoder.
    ax.annotate("", xy=(0.66, 0.42), xytext=(0.52, 0.32),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#7c3aed", linestyle="--"))
    ax.annotate("", xy=(0.66, 0.50), xytext=(0.52, 0.54),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#7c3aed"))
    ax.text(0.59, 0.41, "MSE / cosine /\nrelative L2", ha="center", va="center",
            fontsize=10, color="#6d28d9", fontweight="bold")
    ax.text(0.76, 0.39, "same frozen\nmask decoder", ha="center", va="center",
            fontsize=10, color="#166534", fontweight="bold")
    ax.plot([0.79, 0.84], [0.62, 0.32], color="#166534", lw=1.4, linestyle=":")

    _save_figure(figure, "01_latent_alignment_pipeline", output_dir, dpi)


def _mask_image(mask: np.ndarray, cmap: str = "gray"):
    return np.asarray(mask, dtype=np.float32), cmap, 0.0, 1.0


def _save_native_visuals(
    output_dir: Path,
    sample_index: int,
    image: Image.Image,
    gt: np.ndarray,
    teacher_prediction: np.ndarray,
    student_prediction: np.ndarray,
    cosine_map: np.ndarray,
    student_sam_prediction: np.ndarray | None = None,
) -> None:
    """Save every visual output at exactly the original image resolution."""
    stem = f"sample_{int(sample_index):05d}"
    image.save(output_dir / f"{stem}_image.png")
    masks = [
        ("gt_mask", gt),
        ("teacher_decode", teacher_prediction),
        ("student_decode", student_prediction),
    ]
    if student_sam_prediction is not None:
        masks.append(("student_sam_decode", student_sam_prediction))
    for name, mask in masks:
        mask_array = (np.asarray(mask) >= 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_array, mode="L").save(output_dir / f"{stem}_{name}.png")
    cosine_array = np.clip(np.asarray(cosine_map, dtype=np.float32), 0.0, 1.0)
    Image.fromarray(np.uint8(np.round(cosine_array * 255.0)), mode="L").save(
        output_dir / f"{stem}_cosine_map_gray.png"
    )
    plt.imsave(output_dir / f"{stem}_cosine_map_viridis.png", cosine_array, cmap="viridis", vmin=0.0, vmax=1.0)
    cosine_rgb = np.uint8(np.round(plt.get_cmap("viridis")(cosine_array)[..., :3] * 255.0))
    cosine_color = Image.fromarray(cosine_rgb, mode="RGB")
    Image.blend(image.convert("RGB"), cosine_color, alpha=0.45).save(
        output_dir / f"{stem}_cosine_overlay.png"
    )
    np.save(output_dir / f"{stem}_cosine_map.npy", cosine_array)


def draw_heatmap(
    output_dir: Path,
    dpi: int,
    sample_index: int,
    sample: dict,
    gt: np.ndarray,
    teacher_prediction: np.ndarray,
    student_prediction: np.ndarray,
    cosine_map: np.ndarray,
    sample_cosine: float,
    sample_relative_l2: float,
    sample_iou: float,
    student_sam_prediction: np.ndarray | None = None,
    sample_sam_iou: float | None = None,
) -> None:
    with Image.open(sample["image_path"]) as source:
        image = source.convert("RGB")
    _save_native_visuals(
        output_dir, sample_index, image, gt, teacher_prediction, student_prediction, cosine_map,
        student_sam_prediction,
    )
    with_sam = student_sam_prediction is not None
    panels = 6 if with_sam else 5
    figure, axes = plt.subplots(1, panels, figsize=(3.5 * panels, 3.8))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    gt_image, gt_cmap, gt_vmin, gt_vmax = _mask_image(gt)
    axes[1].imshow(gt_image, cmap=gt_cmap, vmin=gt_vmin, vmax=gt_vmax)
    axes[1].set_title("GT mask")
    teacher_image, teacher_cmap, teacher_vmin, teacher_vmax = _mask_image(teacher_prediction)
    axes[2].imshow(teacher_image, cmap=teacher_cmap, vmin=teacher_vmin, vmax=teacher_vmax)
    axes[2].set_title("Teacher decode")
    student_image, student_cmap, student_vmin, student_vmax = _mask_image(student_prediction)
    axes[3].imshow(student_image, cmap=student_cmap, vmin=student_vmin, vmax=student_vmax)
    axes[3].set_title("Student decode (head)" if with_sam else "Student decode")
    if with_sam:
        axes[4].imshow(np.asarray(student_sam_prediction, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        axes[4].set_title(f"Student decode (SAM)\nIoU={sample_sam_iou:.3f}")
    heat_axis = axes[-1]
    heat = heat_axis.imshow(cosine_map, cmap="viridis", vmin=0.0, vmax=1.0)
    map_height, map_width = np.asarray(cosine_map).shape[-2:]
    heat_axis.set_title(f"Cosine map ({map_width}×{map_height})\nmean={sample_cosine:.3f}")
    figure.colorbar(heat, ax=heat_axis, fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    expression = str(sample.get("expression", ""))
    if len(expression) > 105:
        expression = expression[:102] + "..."
    figure.suptitle(
        f"Latent alignment diagnostic | head IoU={sample_iou:.3f}, relative L2={sample_relative_l2:.3f}\n"
        f"Expression: {expression}", fontsize=12, y=1.04,
    )
    figure.tight_layout()
    _save_figure(figure, f"02_latent_similarity_heatmap_{int(sample_index):05d}", output_dir, dpi)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def _save_quant_panel(output_dir: Path, dpi: int, stem: str, plotter) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    plotter(axis)
    figure.tight_layout()
    _save_figure(figure, stem, output_dir, dpi)


def draw_quantitative(output_dir: Path, dpi: int, rows: Sequence[dict], training_phase: str) -> Dict[str, float]:
    cosine = np.asarray([float(row["cosine_mean"]) for row in rows], dtype=np.float64)
    relative = np.asarray([float(row["relative_l2"]) for row in rows], dtype=np.float64)
    ciou = np.asarray([float(row["head_ciou"]) for row in rows], dtype=np.float64)
    mse = np.asarray([float(row["mse"]) for row in rows], dtype=np.float64)
    summary = {
        "samples": int(len(rows)),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "relative_l2_mean": float(relative.mean()),
        "relative_l2_median": float(np.median(relative)),
        "mse_mean": float(mse.mean()),
        "head_cIoU_mean": float(ciou.mean()),
        "head_Dice_mean": float(np.mean([float(row["head_dice"]) for row in rows])),
        "cosine_vs_cIoU_pearson": _pearson(cosine, ciou),
        "cosine_vs_cIoU_spearman": _spearman(cosine, ciou),
        "relative_l2_vs_cIoU_pearson": _pearson(relative, ciou),
        "relative_l2_vs_cIoU_spearman": _spearman(relative, ciou),
    }

    def plot_cosine_hist(axis) -> None:
        axis.hist(cosine, bins=24, color="#2563eb", alpha=0.82, edgecolor="white")
        axis.axvline(summary["cosine_mean"], color="#b91c1c", linestyle="--", linewidth=1.5,
                     label=f"mean={summary['cosine_mean']:.3f}")
        axis.set_title("Student-Teacher cosine similarity")
        axis.set_xlabel("Mean cosine over 32x32 latent positions")
        axis.set_ylabel("Number of samples")
        axis.legend(frameon=False)

    def plot_relative_hist(axis) -> None:
        axis.hist(relative, bins=24, color="#d97706", alpha=0.82, edgecolor="white")
        axis.axvline(summary["relative_l2_mean"], color="#b91c1c", linestyle="--", linewidth=1.5,
                     label=f"mean={summary['relative_l2_mean']:.3f}")
        axis.set_title("Relative latent L2 error")
        axis.set_xlabel("Mean ||z_S - z_T|| / ||z_T||")
        axis.set_ylabel("Number of samples")
        axis.legend(frameon=False)

    def plot_cosine_ciou(axis) -> None:
        axis.scatter(cosine, ciou, s=12, alpha=0.35, color="#2563eb", edgecolors="none")
        axis.set_title(
            f"Cosine vs decoded-mask IoU\nPearson={summary['cosine_vs_cIoU_pearson']:.3f}, "
            f"Spearman={summary['cosine_vs_cIoU_spearman']:.3f}"
        )
        axis.set_xlabel("Latent cosine similarity")
        axis.set_ylabel("Student head IoU (per sample)")
        axis.set_ylim(0.0, 1.0)

    def plot_relative_ciou(axis) -> None:
        axis.scatter(relative, ciou, s=12, alpha=0.35, color="#d97706", edgecolors="none")
        axis.set_title(
            f"Relative latent error vs IoU\nPearson={summary['relative_l2_vs_cIoU_pearson']:.3f}, "
            f"Spearman={summary['relative_l2_vs_cIoU_spearman']:.3f}"
        )
        axis.set_xlabel("Relative latent L2 error")
        axis.set_ylabel("Student head IoU (per sample)")
        axis.set_ylim(0.0, 1.0)

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    plot_cosine_hist(axes[0, 0])
    plot_relative_hist(axes[0, 1])
    plot_cosine_ciou(axes[1, 0])
    plot_relative_ciou(axes[1, 1])
    phase_label = "Stage 2A / latent warmup" if training_phase == "latent_warmup" else "Stage 2B / task adaptation"
    figure.suptitle(
        f"Validation-set {phase_label} latent alignment (N={len(rows)})",
        fontsize=16, fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "03_latent_similarity_quantitative", output_dir, dpi)
    _save_quant_panel(output_dir, dpi, "03a_cosine_similarity_distribution", plot_cosine_hist)
    _save_quant_panel(output_dir, dpi, "03b_relative_latent_l2_distribution", plot_relative_hist)
    _save_quant_panel(output_dir, dpi, "03c_cosine_vs_head_ciou", plot_cosine_ciou)
    _save_quant_panel(output_dir, dpi, "03d_relative_l2_vs_head_ciou", plot_relative_ciou)
    return summary


HEAD_COLOR = "#eb6834"  # image-free head
SAM_COLOR = "#2a78d6"   # image-conditioned SAM path (ours)


def _overlap(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Intersection and union pixel counts of two boolean masks."""
    return (
        float((prediction & target).sum(dtype=torch.float64).item()),
        float((prediction | target).sum(dtype=torch.float64).item()),
    )


def _path_scores(rows: Sequence[dict], prefix: str) -> Dict[str, float]:
    """cIoU = sum(I) / sum(U); gIoU = mean per-sample IoU (empty union counts as 1)."""
    inter = np.asarray([float(row[f"{prefix}_inter"]) for row in rows], dtype=np.float64)
    union = np.asarray([float(row[f"{prefix}_union"]) for row in rows], dtype=np.float64)
    per_sample = np.where(union > 0.0, inter / np.maximum(union, 1.0), 1.0)
    return {
        "cIoU": float(100.0 * inter.sum() / max(union.sum(), 1.0)),
        "gIoU": float(100.0 * per_sample.mean()),
    }


def draw_decode_comparison(output_dir: Path, dpi: int, rows: Sequence[dict], training_phase: str) -> Dict[str, object]:
    """Same Student latent, two decoders, scored at the original resolution."""
    head = _path_scores(rows, "head_orig")
    sam = _path_scores(rows, "sam_orig")
    head_iou = np.asarray([float(row["head_orig_iou"]) for row in rows], dtype=np.float64)
    sam_iou = np.asarray([float(row["sam_orig_iou"]) for row in rows], dtype=np.float64)
    delta = sam_iou - head_iou
    comparison = {
        "image_free_head": head,
        "sam_path": sam,
        "delta_sam_minus_head": {
            "cIoU": sam["cIoU"] - head["cIoU"],
            "gIoU": sam["gIoU"] - head["gIoU"],
        },
        "samples_sam_better": int((delta > 0.0).sum()),
        "samples_head_better": int((delta < 0.0).sum()),
        "samples_tied": int((delta == 0.0).sum()),
        "note": "Both paths decode the identical Student latent; IoU is computed after restoring each mask to the original image resolution.",
    }

    phase_label = "Stage 2A / latent warmup" if training_phase == "latent_warmup" else "Stage 2B / task adaptation"
    figure, (bar_axis, scatter_axis) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    metrics = ["cIoU", "gIoU"]
    positions = np.arange(len(metrics), dtype=np.float64)
    width = 0.36
    for offset, scores, color, label in (
        (-width / 2.0 - 0.01, head, HEAD_COLOR, "Image-free head"),
        (width / 2.0 + 0.01, sam, SAM_COLOR, "SAM path (ours)"),
    ):
        values = [scores[name] for name in metrics]
        bars = bar_axis.bar(positions + offset, values, width=width, color=color, label=label, zorder=2)
        for bar, value in zip(bars, values):
            bar_axis.text(bar.get_x() + bar.get_width() / 2.0, value + 0.6, f"{value:.2f}",
                          ha="center", va="bottom", fontsize=10, color="#1f2937")
    lowest = min(min(head.values()), min(sam.values()))
    bar_axis.set_ylim(max(0.0, np.floor(lowest / 10.0) * 10.0 - 10.0), 100.0)
    bar_axis.set_xticks(positions, metrics)
    bar_axis.set_ylabel("Score (%)")
    bar_axis.set_title("Dataset-level score, same Student latent")
    bar_axis.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    bar_axis.legend(frameon=False, loc="upper left")
    for side in ("top", "right"):
        bar_axis.spines[side].set_visible(False)

    scatter_axis.scatter(head_iou, sam_iou, s=10, alpha=0.30, color=SAM_COLOR, edgecolors="none", zorder=2)
    scatter_axis.plot([0.0, 1.0], [0.0, 1.0], color="#6b7280", linestyle="--", linewidth=1.0, zorder=1)
    scatter_axis.set_xlim(0.0, 1.0)
    scatter_axis.set_ylim(0.0, 1.0)
    scatter_axis.set_aspect("equal")
    scatter_axis.set_xlabel("Image-free head IoU")
    scatter_axis.set_ylabel("SAM path IoU")
    scatter_axis.set_title(
        f"Per-sample IoU (above diagonal: SAM better)\n"
        f"SAM better {comparison['samples_sam_better']} · head better {comparison['samples_head_better']}"
        f" · tied {comparison['samples_tied']}"
    )
    for side in ("top", "right"):
        scatter_axis.spines[side].set_visible(False)
    figure.suptitle(f"Decoding path comparison | {phase_label} (N={len(rows)})", fontsize=14, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, "04_decode_path_comparison", output_dir, dpi)
    return comparison


def _sample_metrics(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Tuple[List[dict], np.ndarray]:
    student = student_latent.float()
    teacher = teacher_latent.float()
    difference = student - teacher
    cosine_map = F.cosine_similarity(student, teacher, dim=1, eps=1e-6)
    relative_map = difference.square().sum(dim=1).sqrt() / teacher.square().sum(dim=1).sqrt().clamp_min(1e-6)
    mse = difference.square().flatten(1).mean(dim=1)
    student_pred = student_logits.float().sigmoid() >= float(threshold)
    teacher_pred = teacher_logits.float().sigmoid() >= float(threshold)
    target_bool = target.float() >= 0.5
    rows = []
    for index in range(student.shape[0]):
        pred = student_pred[index, 0]
        ref = target_bool[index, 0]
        inter = float((pred & ref).sum().item())
        union = float((pred | ref).sum().item())
        pred_area = float(pred.sum().item())
        ref_area = float(ref.sum().item())
        iou = 1.0 if union == 0.0 else inter / union
        dice = 1.0 if pred_area + ref_area == 0.0 else 2.0 * inter / (pred_area + ref_area)
        rows.append({
            "cosine_mean": float(cosine_map[index].mean().item()),
            "relative_l2": float(relative_map[index].mean().item()),
            "mse": float(mse[index].item()),
            "head_ciou": iou,
            "head_dice": dice,
            "teacher_head_ciou": 1.0 if ((teacher_pred[index, 0] | ref).sum().item() == 0) else float(((teacher_pred[index, 0] & ref).sum().item()) / ((teacher_pred[index, 0] | ref).sum().item())),
        })
    return rows, cosine_map.detach().cpu().numpy()


def _parse_visual_indices(raw: str, fallback: int, count: int, visual_count: int) -> List[int]:
    if not str(raw or "").strip() and int(visual_count) > 0:
        return list(range(min(int(visual_count), count)))
    values = str(raw or "").replace(";", ",").split(",")
    parsed: List[int] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid visual index: {value!r}") from error
        if 0 <= index < count and index not in parsed:
            parsed.append(index)
    if not parsed:
        index = min(max(0, int(fallback)), count - 1)
        parsed.append(index)
    return parsed


def _restore_to_original(model, masks: torch.Tensor, sam_input_size: torch.Tensor,
                         original_size: torch.Tensor, device: torch.device) -> np.ndarray:
    """Undo SAM's padded square geometry and return an HxW numpy array."""
    restored = model.teacher.sam.postprocess_masks(
        masks.to(device=device, dtype=torch.float32),
        tuple(map(int, sam_input_size.detach().cpu().tolist())),
        tuple(map(int, original_size.detach().cpu().tolist())),
    )
    return restored[0, 0].detach().cpu().numpy()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


@torch.no_grad()
def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, FigureArguments))
    model_args, data_args, figure_args = parser.parse_args_into_dataclasses()
    if int(figure_args.max_samples) < 0:
        raise ValueError("--max_samples must be zero or positive")
    if not model_args.teacher_checkpoint or not os.path.isfile(model_args.teacher_checkpoint):
        raise FileNotFoundError("--teacher_checkpoint must point to a Stage-1 Teacher checkpoint")
    if not model_args.student_checkpoint or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("--student_checkpoint must point to a Stage-2 Student checkpoint")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()

    eval_data_args = DataArguments(**{
        **data_args.__dict__,
        "train_split": figure_args.split,
        "train_subset_ratio": 1.0,
        "synthetic_negative_ratio": 0.0,
        "load_sam_images": False,
        "hard_negative_cache": "",
        "hard_negative_max": 0,
    })
    # HfArgumentParser versions differ in how they materialize boolean CLI
    # values.  Parse explicitly so ``--decode_comparison False`` can never be
    # treated as truthy merely because it is a non-empty string.
    compare_value = figure_args.decode_comparison
    compare = str(compare_value).strip().lower() in {"1", "true", "yes", "y", "on"}
    print(f"decode_comparison enabled: {compare} (raw={compare_value!r})", flush=True)
    eval_data_args.load_sam_images = compare
    dataset = QwenReferDataset(eval_data_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)
    count = len(dataset) if int(figure_args.max_samples) == 0 else min(len(dataset), int(figure_args.max_samples))
    if count == 0:
        raise RuntimeError("The selected split is empty")
    output_dir = Path(figure_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_pipeline(output_dir, int(figure_args.dpi), str(model_args.training_phase))

    rows: List[dict] = []
    visual_indices = _parse_visual_indices(
        figure_args.visual_indices, figure_args.visual_index, count, figure_args.visual_count,
    )
    visual_index_set = set(visual_indices)
    saved_visuals: List[int] = []
    for index in range(count):
        sample = dataset[index]
        batch = collator([sample])
        qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
        sam_input_sizes = batch["sam_input_sizes"].to(device)
        target_latent, mask256 = model.teacher.encode_gt(batch["gt_masks"], sam_input_sizes)
        student_latent = model.predict_latent(qwen_inputs)
        student_head = model.teacher.decode_mask_head(student_latent)
        teacher_head = model.teacher.decode_mask_head(target_latent)
        sample_rows, cosine_map = _sample_metrics(
            student_latent, target_latent, student_head, teacher_head,
            mask256.to(device=device), float(figure_args.threshold),
        )
        row = sample_rows[0]
        original_size = batch["original_sizes"][0]
        sam_input_size = batch["sam_input_sizes"][0]
        student_sam_original = None
        if compare:
            # Same latent, both decoders, both scored at the original resolution
            # with the same threshold the main eval script uses.
            gt_bool = batch["gt_masks"][0][0].to(device=device) >= 0.5
            head_probability = torch.from_numpy(_restore_to_original(
                model, student_head, sam_input_size, original_size, device,
            )).to(device).sigmoid()
            sam_features = model.teacher.encode_sam_images(batch["sam_images"])
            sam_masks256 = model.decode_with_sam_features(student_latent, sam_features)
            sam_logits = model.original_logits_from_sizes(
                sam_masks256, sam_input_sizes, batch["original_sizes"].to(device),
            )[0]
            sam_probability = sam_logits[0, 0].float().sigmoid()
            for prefix, probability in (("head_orig", head_probability), ("sam_orig", sam_probability)):
                prediction = probability >= float(figure_args.threshold)
                inter, union = _overlap(prediction, gt_bool)
                row[f"{prefix}_inter"] = inter
                row[f"{prefix}_union"] = union
                row[f"{prefix}_iou"] = 1.0 if union == 0.0 else inter / union
            student_sam_original = (sam_probability >= float(figure_args.threshold)).cpu().numpy()
        row.update({
            "index": int(index),
            "reference_key": str(sample.get("reference_key", index)),
            "image_path": os.path.abspath(sample["image_path"]),
            "expression": str(sample.get("expression", "")),
        })
        rows.append(row)
        if index in visual_index_set:
            gt_original = batch["gt_masks"][0][0].detach().cpu().numpy() >= 0.5
            # The head emits logits; threshold probabilities, as the metrics do.
            teacher_original = _sigmoid(_restore_to_original(
                model, teacher_head, sam_input_size, original_size, device,
            )) >= float(figure_args.threshold)
            student_original = _sigmoid(_restore_to_original(
                model, student_head, sam_input_size, original_size, device,
            )) >= float(figure_args.threshold)
            cosine_tensor = torch.from_numpy(cosine_map[0]).unsqueeze(0).unsqueeze(0)
            cosine_original = _restore_to_original(
                model, cosine_tensor, sam_input_size, original_size, device,
            )
            draw_heatmap(
                output_dir, int(figure_args.dpi), index, sample, gt_original,
                teacher_original, student_original, cosine_original,
                float(row["cosine_mean"]), float(row["relative_l2"]), float(row["head_ciou"]),
                student_sam_original,
                float(row["sam_orig_iou"]) if compare else None,
            )
            saved_visuals.append(index)
        if (index + 1) % 50 == 0 or index + 1 == count:
            print(f"latent alignment figures: {index + 1}/{count}")

    if not saved_visuals:
        raise RuntimeError("Could not collect the requested visual samples")
    summary = draw_quantitative(output_dir, int(figure_args.dpi), rows, str(model_args.training_phase))
    # head_cIoU_mean is a mean of per-sample IoU on the 256 canvas (a gIoU);
    # keep the old key for compatibility and add an accurately named alias.
    summary["head_IoU_mean_256"] = summary["head_cIoU_mean"]
    if compare:
        summary["decode_comparison"] = draw_decode_comparison(
            output_dir, int(figure_args.dpi), rows, str(model_args.training_phase),
        )
    summary.update({
        "model": "llava6-3-8",
        "training_phase": str(model_args.training_phase),
        "split": str(figure_args.split),
        "qwen_image_size": int(model_args.qwen_image_size),
        "threshold": float(figure_args.threshold),
        "decode_comparison_enabled": bool(compare),
        "visual_samples_saved": saved_visuals,
        "teacher_checkpoint": os.path.abspath(model_args.teacher_checkpoint),
        "student_checkpoint": os.path.abspath(model_args.student_checkpoint),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    with open(output_dir / "latent_alignment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(output_dir / "latent_alignment_samples.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved figures and metrics to {output_dir}")


if __name__ == "__main__":
    main()





