"""Causal readout-swap evaluation for llava6-3-8.

For two expressions referring to different objects in the same image, this
script evaluates A, B, A with B's projected readout, and B with A's projected
readout.  The swapped pass keeps the original sample's image tokens, text
states, SAM image, and decoder; only the 16x16 readout tensor is replaced.
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw

from PLAS.train.train_qwen_dense_latent_refseg import (
    DataArguments, ModelArguments, QwenCollator, QwenReferDataset, build_student,
)


@dataclass
class SwapEvalArguments:
    output_dir: str = "./PLAS/outputs/readout_swap"
    split: str = "val"
    threshold: float = 0.55
    max_pairs: int = 200
    correct_iou: float = 0.50
    decode_path: str = "sam"


def iou(prediction, target):
    prediction = _mask2d(prediction)
    target = _mask2d(target)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(1, union))


def _mask2d(mask):
    """Normalize [1,H,W], [H,W], or torch masks to a 2-D boolean array."""
    value = np.asarray(mask)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    while value.ndim > 2 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2-D mask after squeezing singleton axes, got {value.shape}.")
    return value.astype(bool)


def overlay(image, mask, color):
    value = np.asarray(image).astype(np.float32)
    mask = _mask2d(mask)
    value[mask] = 0.55 * value[mask] + 0.45 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(value.clip(0, 255).astype(np.uint8))


def panel(image, masks, labels):
    tiles = []
    for mask, label in zip(masks, labels):
        tile = overlay(image, mask, (255, 50, 50))
        canvas = Image.new("RGB", (tile.width, tile.height + 24), "white")
        canvas.paste(tile, (0, 24))
        ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
        tiles.append(canvas)
    output = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    offset = 0
    for tile in tiles:
        output.paste(tile, (offset, 0))
        offset += tile.width
    return output


def make_pairs(dataset, maximum):
    by_image = {}
    for index, record in enumerate(dataset.records):
        ann_key = tuple(record[6])
        if not ann_key:
            continue
        by_image.setdefault(int(record[5]), {}).setdefault(ann_key, index)
    pairs = []
    for image_id in sorted(by_image):
        representatives = list(by_image[image_id].values())
        pairs.extend(combinations(representatives, 2))
        # A non-positive maximum means "use every valid pair".  Only apply
        # the early stop and slicing for an explicit positive cap.
        if maximum > 0 and len(pairs) >= maximum:
            return pairs[:maximum]
    return pairs if maximum <= 0 else pairs[:maximum]


def sample_batch(dataset, collator, index, device):
    sample = dataset[index]
    batch = collator([sample])
    qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
    return sample, batch, qwen_inputs


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, SwapEvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(model_args.teacher_checkpoint) or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("Readout swapping requires teacher_checkpoint and student_checkpoint.")

    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()
    eval_data_args = DataArguments(**{
        **data_args.__dict__, "train_split": eval_args.split,
        "train_subset_ratio": 1.0, "synthetic_negative_ratio": 0.0,
    })
    dataset = QwenReferDataset(eval_data_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)
    pairs = make_pairs(dataset, int(eval_args.max_pairs))
    if not pairs:
        raise RuntimeError("No same-image pairs with distinct target annotations were found.")
    os.makedirs(eval_args.output_dir, exist_ok=True)

    records = []
    for pair_index, (index_a, index_b) in enumerate(pairs):
        sample_a, batch_a, inputs_a = sample_batch(dataset, collator, index_a, device)
        sample_b, batch_b, inputs_b = sample_batch(dataset, collator, index_b, device)
        sizes_a = batch_a["sam_input_sizes"].to(device)
        original_a = batch_a["original_sizes"].to(device)
        sizes_b = batch_b["sam_input_sizes"].to(device)
        original_b = batch_b["original_sizes"].to(device)

        logits_a, _, probe_a = model.predict(
            inputs_a, batch_a["sam_images"], sizes_a, original_a,
            decode_path=eval_args.decode_path, return_probe_features=True,
        )
        logits_b, _, probe_b = model.predict(
            inputs_b, batch_b["sam_images"], sizes_b, original_b,
            decode_path=eval_args.decode_path, return_probe_features=True,
        )
        logits_a_from_b, _ = model.predict(
            inputs_a, batch_a["sam_images"], sizes_a, original_a,
            decode_path=eval_args.decode_path, override_readout=probe_b["readout"],
        )
        logits_b_from_a, _ = model.predict(
            inputs_b, batch_b["sam_images"], sizes_b, original_b,
            decode_path=eval_args.decode_path, override_readout=probe_a["readout"],
        )

        pred_a = _mask2d(logits_a[0].sigmoid().cpu().numpy() >= eval_args.threshold)
        pred_b = _mask2d(logits_b[0].sigmoid().cpu().numpy() >= eval_args.threshold)
        pred_a_from_b = _mask2d(logits_a_from_b[0].sigmoid().cpu().numpy() >= eval_args.threshold)
        pred_b_from_a = _mask2d(logits_b_from_a[0].sigmoid().cpu().numpy() >= eval_args.threshold)
        gt_a = batch_a["gt_masks"][0].squeeze().cpu().numpy() >= 0.5
        gt_b = batch_b["gt_masks"][0].squeeze().cpu().numpy() >= 0.5

        outputs = {
            "normal_A": pred_a, "normal_B": pred_b,
            "A_readout_B": pred_a_from_b, "B_readout_A": pred_b_from_a,
        }
        scores = {
            name: {"target_A_iou": iou(pred, gt_a), "target_B_iou": iou(pred, gt_b),
                   "preference_B_minus_A": iou(pred, gt_b) - iou(pred, gt_a)}
            for name, pred in outputs.items()
        }
        image_path = sample_a["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        stem = f"{pair_index:05d}_image{int(dataset.records[index_a][5])}"
        panel_image = panel(
            image, [gt_a, gt_b, pred_a, pred_b, pred_a_from_b, pred_b_from_a],
            ["GT A", "GT B", "A", "B", "A + R_B", "B + R_A"],
        )
        panel_path = os.path.join(eval_args.output_dir, stem + "_panel.jpg")
        panel_image.save(panel_path, quality=95)
        for name, mask in outputs.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                os.path.join(eval_args.output_dir, f"{stem}_{name}.png")
            )
        records.append({
            "pair_index": pair_index, "image_id": int(dataset.records[index_a][5]),
            "index_A": index_a, "index_B": index_b,
            "expression_A": sample_a["expression"], "expression_B": sample_b["expression"],
            "reference_key_A": sample_a["reference_key"], "reference_key_B": sample_b["reference_key"],
            "panel": os.path.abspath(panel_path), "scores": scores,
            "both_correct": (
                scores["normal_A"]["target_A_iou"] >= eval_args.correct_iou and
                scores["normal_B"]["target_B_iou"] >= eval_args.correct_iou
            ),
        })
        print(f"[{pair_index + 1}/{len(pairs)}] {sample_a['expression']}  <->  {sample_b['expression']}")

    def summarize(rows):
        if not rows:
            return {"count": 0}
        keys = ("normal_A", "normal_B", "A_readout_B", "B_readout_A")
        return {
            "count": len(rows),
            **{key: {
                "target_A_iou": float(np.mean([row["scores"][key]["target_A_iou"] for row in rows])),
                "target_B_iou": float(np.mean([row["scores"][key]["target_B_iou"] for row in rows])),
                "preference_B_minus_A": float(np.mean([row["scores"][key]["preference_B_minus_A"] for row in rows])),
            } for key in keys},
        }

    summary = {
        "split": eval_args.split, "threshold": eval_args.threshold,
        "pairs": len(records), "correct_iou": eval_args.correct_iou,
        "all_pairs": summarize(records),
        "both_normal_predictions_correct": summarize([row for row in records if row["both_correct"]]),
        "swap_A_to_B_preference_gain": float(np.mean([
            row["scores"]["A_readout_B"]["preference_B_minus_A"] - row["scores"]["normal_A"]["preference_B_minus_A"]
            for row in records
        ])),
        "swap_B_to_A_preference_gain": float(np.mean([
            row["scores"]["B_readout_A"]["preference_B_minus_A"] - row["scores"]["normal_B"]["preference_B_minus_A"]
            for row in records
        ])),
        "swap_A_to_B_success_rate": float(np.mean([
            row["scores"]["A_readout_B"]["preference_B_minus_A"] >
            row["scores"]["normal_A"]["preference_B_minus_A"]
            for row in records
        ])),
        "swap_B_to_A_success_rate": float(np.mean([
            row["scores"]["B_readout_A"]["preference_B_minus_A"] <
            row["scores"]["normal_B"]["preference_B_minus_A"]
            for row in records
        ])),
        "swap_A_to_B_flip_rate": float(np.mean([
            row["scores"]["normal_A"]["preference_B_minus_A"] < 0.0 and
            row["scores"]["A_readout_B"]["preference_B_minus_A"] > 0.0
            for row in records
        ])),
        "swap_B_to_A_flip_rate": float(np.mean([
            row["scores"]["normal_B"]["preference_B_minus_A"] > 0.0 and
            row["scores"]["B_readout_A"]["preference_B_minus_A"] < 0.0
            for row in records
        ])),
    }
    with open(os.path.join(eval_args.output_dir, "readout_swap_results.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "pairs": records}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()







