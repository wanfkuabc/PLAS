"""Teacher-oracle evaluation on ReasonSeg.

The ground-truth mask is encoded by the Stage-I Teacher and decoded through
the SAM-H prompt adapter and mask decoder.  This is a diagnostic ceiling, not
an end-to-end model result: it removes the Qwen/student language-to-latent
step while keeping the SAM image encoder, prompt adapter, mask decoder, and
post-processing unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from PLAS.eval.eval_qwen_dense_latent_reasonseg import (
    find_image_path,
    polygon_mask,
    resolve_split_dir,
    save_visual,
)
from PLAS.model.mask_latent_teacher import MaskLatentTeacher
from PLAS.train.qwen_sam_geometry import sam_image_input, sam_resize_size


class TeacherReasonSegDataset:
    def __init__(self, root: str, split: str):
        if split not in {"val", "test"}:
            raise ValueError("ReasonSeg supports split=val or split=test.")
        directory = resolve_split_dir(root, split)
        self.records = []
        for annotation_path in sorted(directory.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            image_path = find_image_path(annotation_path, annotation)
            self.records.append((annotation_path, annotation, image_path))
        if not self.records:
            raise RuntimeError(f"No ReasonSeg JSON annotations found in {directory}")
        print(f"Teacher-oracle ReasonSeg set: {len(self.records)} images, split={split}, directory={directory}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        annotation_path, annotation, image_path = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        annotation_height = annotation.get("imageHeight")
        annotation_width = annotation.get("imageWidth")
        target = polygon_mask(
            annotation.get("shapes", []), "target", image.height, image.width,
            annotation_height, annotation_width,
        )
        ignore = polygon_mask(
            annotation.get("shapes", []), "ignore", image.height, image.width,
            annotation_height, annotation_width,
        )
        texts = annotation.get("text", [])
        if isinstance(texts, str):
            expression = texts
        else:
            expression = next((item for item in texts if isinstance(item, str) and item.strip()), "")
        return {
            "reference_key": annotation_path.stem,
            "expression": expression,
            "image_path": str(image_path),
            "sam_images": sam_image_input(image),
            "gt_masks": torch.from_numpy(target[None].astype(np.float32)),
            "ignore_mask": torch.from_numpy(ignore[None]),
            "original_sizes": torch.tensor([image.height, image.width], dtype=torch.long),
            "sam_input_sizes": torch.tensor(sam_resize_size(image), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the llava6-3-8 Stage-I Teacher oracle on ReasonSeg.")
    parser.add_argument("--sam_path", required=True, help="SAM-H checkpoint used by the Teacher.")
    parser.add_argument("--teacher_checkpoint", required=True, help="Stage-I mask_latent_teacher.pt checkpoint.")
    parser.add_argument("--sam_model_type", default="vit_h")
    parser.add_argument("--reasonseg_root", required=True, help="Directory containing ReasonSeg val/ and test/.")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output_dir", default="./PLAS/outputs/reasonseg_teacher_oracle")
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the complete split.")
    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--visual_max_samples", type=int, default=256)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.sam_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_path}")
    if not os.path.isfile(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    requested_device = str(args.device).strip().lower()
    if requested_device == "cuda":
        # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0.
        requested_device = "cuda:0"
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    teacher = MaskLatentTeacher(args.sam_path, args.sam_model_type)
    teacher.load_teacher(args.teacher_checkpoint)
    teacher.to(device=device).eval()

    dataset = TeacherReasonSegDataset(args.reasonseg_root, args.split)
    total = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    image_iou_sum = 0.0
    intersection_sum = 0.0
    union_sum = 0.0
    empty_count = 0
    empty_correct = 0

    for index in range(total):
        sample = dataset[index]
        gt_masks = [sample["gt_masks"].to(device=device)]
        sam_input_sizes = sample["sam_input_sizes"].unsqueeze(0).to(device=device)
        original_sizes = sample["original_sizes"].unsqueeze(0).to(device=device)

        # Oracle path: GT mask -> Teacher latent -> dense SAM prompt -> SAM-H mask.
        latent, _ = teacher.encode_gt(gt_masks, sam_input_sizes)
        sam_features = teacher.encode_sam_images([sample["sam_images"]])
        masks256 = teacher.decode_latent(latent, sam_features)
        logits = teacher.original_logits(masks256, sam_input_sizes, original_sizes)[0]
        prediction = logits.sigmoid() >= float(args.threshold)

        target = sample["gt_masks"].to(device=device, dtype=torch.bool)
        ignore = sample["ignore_mask"].to(device=device, dtype=torch.bool)
        prediction = prediction & ~ignore
        target = target & ~ignore
        intersection = float((prediction & target).sum().item())
        union = float((prediction | target).sum().item())
        iou = 1.0 if union == 0.0 else intersection / union
        image_iou_sum += iou
        intersection_sum += intersection
        union_sum += union

        target_is_empty = not bool(target.any())
        if target_is_empty:
            empty_count += 1
            empty_correct += int(not bool(prediction.any()))

        rows.append({
            "index": index,
            "reference_key": sample["reference_key"],
            "image_path": sample["image_path"],
            "expression": sample["expression"],
            "iou": iou * 100.0,
            "intersection": intersection,
            "union": union,
            "target_area": int(target.sum().item()),
            "prediction_area": int(prediction.sum().item()),
            "target_empty": int(target_is_empty),
        })

        if args.save_visuals and (args.visual_max_samples < 0 or index < args.visual_max_samples):
            save_visual(output_dir, index, 0, sample, prediction, target, ignore)

        if (index + 1) % 20 == 0 or index + 1 == total:
            print(f"[{index + 1}/{total}] gIoU-so-far={100.0 * image_iou_sum / (index + 1):.4f}", flush=True)

    with (output_dir / "teacher_oracle_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": "llava6-3-8-teacher-oracle",
        "split": args.split,
        "samples": total,
        "threshold": float(args.threshold),
        "oracle_path": "ground-truth mask -> Teacher mask encoder -> prompt adapter -> SAM-H decoder",
        "gIoU_percent": 100.0 * image_iou_sum / max(1, total),
        "cIoU_percent": 100.0 * intersection_sum / max(1.0, union_sum),
        "empty_count": empty_count,
        "N_acc_percent": 100.0 * empty_correct / max(1, empty_count),
        "sam_path": str(Path(args.sam_path).resolve()),
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
    }
    with (output_dir / "teacher_oracle_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {output_dir / 'teacher_oracle_summary.json'}")
    print(f"Saved per-sample rows: {output_dir / 'teacher_oracle_samples.csv'}")


if __name__ == "__main__":
    main()





