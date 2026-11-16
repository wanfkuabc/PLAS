"""Benchmark llava6-3-8 quality and inference efficiency.

This script uses the same construction and dataset path as
``eval_qwen_dense_latent_refseg.py`` and reports both:

* ``model_latency_ms``: model.predict only, with CUDA synchronization;
* ``end_to_end_latency_ms``: dataset decoding + multimodal preprocessing +
  model.predict + thresholding/metric bookkeeping.

The summary is written to ``latency_benchmark.json`` and one row per measured
sample is written to ``latency_samples.csv``.  Model loading and warm-up are
never included in the reported latency statistics.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch
import transformers

from latency_components import timed_model_components

from PLAS.train.train_qwen_dense_latent_refseg import (
    DataArguments,
    ModelArguments,
    QwenCollator,
    QwenReferDataset,
    build_student,
)


@dataclass
class BenchmarkArguments:
    output_dir: str = "./PLAS/outputs/benchmark"
    split: str = "val"
    threshold: float = 0.55
    warmup: int = 20
    max_samples: int = 0
    seed: int = 3407
    exclude_data_loading: bool = False
    save_sample_rows: bool = True
    # "sam" runs the deployment path: latent -> dense prompt -> SAM ViT-H
    # encoder + mask decoder.  "head" runs the Stage-I image-free mask head and
    # never touches the SAM image encoder.  Both decode the same latent, so the
    # gap between the two summaries is exactly the cost of the SAM stage.
    decode_path: str = "sam"


def initialize_device() -> Tuple[int, torch.device]:
    """Return local rank and device; distributed launch is intentionally rejected."""
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        raise RuntimeError(
            "Latency benchmark must run on one process. Do not launch it with torchrun; "
            "use CUDA_VISIBLE_DEVICES to select one GPU."
        )
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return 0, device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarize_latency(values: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)) if values else float("nan"),
        "std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0 if values else float("nan"),
        "median_ms": float(statistics.median(values)) if values else float("nan"),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "min_ms": float(min(values)) if values else float("nan"),
        "max_ms": float(max(values)) if values else float("nan"),
        "fps_from_mean": float(1000.0 / statistics.fmean(values)) if values else float("nan"),
    }


def prepare_sample(dataset, collator, index: int, device: torch.device):
    sample = dataset[index]
    batch = collator([sample])
    qwen_inputs = {key: value.to(device) for key, value in batch["qwen_inputs"].items()}
    sam_input_sizes = batch["sam_input_sizes"].to(device)
    original_sizes = batch["original_sizes"].to(device)
    return sample, batch, qwen_inputs, sam_input_sizes, original_sizes


@torch.no_grad()
def run_model(model, prepared, device: torch.device, decode_path: str = "sam"):
    synchronize(device)
    logits, latent, components = timed_model_components(
        model, prepared, device, synchronize, decode_path=decode_path,
    )
    probability = logits[0].sigmoid()
    synchronize(device)
    return components["total_model"], logits[0], probability, latent, components


def sample_quality(probability, target, threshold: float):
    prediction = probability >= float(threshold)
    target = target.to(device=prediction.device, dtype=torch.bool)
    intersection = (prediction & target).sum(dtype=torch.float64)
    union = (prediction | target).sum(dtype=torch.float64)
    target_empty = not bool(target.any())
    prediction_empty = not bool(prediction.any())
    if target_empty:
        iou = 1.0 if prediction_empty else 0.0
    else:
        iou = float((intersection / union.clamp_min(1.0)).item())
    return prediction, float(intersection.item()), float(union.item()), iou, target_empty


@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, BenchmarkArguments))
    model_args, data_args, benchmark_args = parser.parse_args_into_dataclasses()
    _, device = initialize_device()
    if not model_args.teacher_checkpoint or not os.path.isfile(model_args.teacher_checkpoint):
        raise FileNotFoundError("--teacher_checkpoint must point to a Stage-1 Teacher checkpoint.")
    if not model_args.student_checkpoint or not os.path.isfile(model_args.student_checkpoint):
        raise FileNotFoundError("--student_checkpoint must point to a Student checkpoint.")
    if int(benchmark_args.warmup) < 0:
        raise ValueError("--warmup must be non-negative.")
    if int(benchmark_args.max_samples) < 0:
        raise ValueError("--max_samples must be zero or positive.")

    torch.manual_seed(int(benchmark_args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(benchmark_args.seed))

    processor, model = build_student(model_args)
    model.load_student(model_args.student_checkpoint)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.restore_fp32_modules()

    eval_data_args = DataArguments(**{
        **data_args.__dict__,
        "train_split": benchmark_args.split,
        "train_subset_ratio": 1.0,
        "synthetic_negative_ratio": 0.0,
    })
    dataset = QwenReferDataset(eval_data_args, processor, model_args.qwen_image_size)
    collator = QwenCollator(processor)
    if len(dataset) == 0:
        raise RuntimeError("The selected split is empty.")
    measured_count = len(dataset) if benchmark_args.max_samples == 0 else min(
        len(dataset), int(benchmark_args.max_samples)
    )

    # Warm-up on one prepared example.  Warm-up is repeated so model loading,
    # CUDA kernel selection, and allocator growth do not enter the statistics.
    warmup_prepared = prepare_sample(dataset, collator, 0, device)
    for _ in range(int(benchmark_args.warmup)):
        run_model(model, warmup_prepared, device, benchmark_args.decode_path)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_latencies: List[float] = []
    end_to_end_latencies: List[float] = []
    component_latencies: Dict[str, List[float]] = {}
    rows: List[Dict[str, object]] = []
    iou_sum = 0.0
    intersection_sum = 0.0
    union_sum = 0.0
    positive_iou_sum = 0.0
    positive_count = 0
    empty_correct = 0
    empty_count = 0

    for index in range(measured_count):
        if not benchmark_args.exclude_data_loading:
            synchronize(device)
            end_to_end_start = time.perf_counter()
            prepared = prepare_sample(dataset, collator, index, device)
        else:
            prepared = prepare_sample(dataset, collator, index, device)
            synchronize(device)
            end_to_end_start = time.perf_counter()
        sample, batch, _, _, _ = prepared
        model_ms, _, probability, _, components = run_model(model, prepared, device, benchmark_args.decode_path)
        prediction, intersection, union, iou, target_empty = sample_quality(
            probability, batch["gt_masks"][0], benchmark_args.threshold
        )
        synchronize(device)
        end_to_end_ms = 1000.0 * (time.perf_counter() - end_to_end_start)

        model_latencies.append(model_ms)
        end_to_end_latencies.append(end_to_end_ms)
        for name, value in components.items():
            component_latencies.setdefault(name, []).append(float(value))
        iou_sum += iou
        if target_empty:
            empty_count += 1
            empty_correct += int(not bool(prediction.any()))
        else:
            positive_count += 1
            positive_iou_sum += iou
            intersection_sum += intersection
            union_sum += union

        if benchmark_args.save_sample_rows:
            rows.append({
                "index": index,
                "reference_key": sample.get("reference_key", ""),
                "image_path": os.path.abspath(sample["image_path"]),
                "expression": sample.get("expression", ""),
                "model_latency_ms": model_ms,
                "end_to_end_latency_ms": end_to_end_ms,
                **{f"{name}_ms": float(value) for name, value in components.items()},
                "sample_iou": iou,
                "target_empty": target_empty,
            })

    peak_allocated = float("nan")
    peak_reserved = float("nan")
    if device.type == "cuda":
        peak_allocated = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

    os.makedirs(benchmark_args.output_dir, exist_ok=True)
    summary = {
        "model": "llava6-3-8",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "split": benchmark_args.split,
        "samples": measured_count,
        "dataset_size": len(dataset),
        "warmup_iterations": int(benchmark_args.warmup),
        "batch_size": 1,
        "precision": "bfloat16 model with fp32 prompt modules",
        "qwen_image_size": int(model_args.qwen_image_size),
        "threshold": float(benchmark_args.threshold),
        "decode_path": str(benchmark_args.decode_path).lower(),
        "includes_data_loading": not bool(benchmark_args.exclude_data_loading),
        "timing_definition": {
            "qwen_prefill": "Qwen3-VL forward with output_hidden_states=True; includes image and readout tokens",
            "readout_fusion": "layer fusion, token extraction, visual pairing, text-spatial attention",
            "latent_adapter": "spatial blocks, upsampling, DetailFusion, latent projection",
            "sam_image_encoder": "SAM ViT-H image encoder",
            "mask_decoder": "latent-to-prompt calibration plus SAM mask decoder",
            "postprocess": "resize logits to original image size",
        },
        "latency_ms": {
            "model_predict": summarize_latency(model_latencies),
            "end_to_end": summarize_latency(end_to_end_latencies),
            "components": {name: summarize_latency(values) for name, values in component_latencies.items()},
        },
        "quality": {
            "gIoU_percent": 100.0 * iou_sum / max(1, measured_count),
            "positive_gIoU_percent": 100.0 * positive_iou_sum / max(1, positive_count),
            "cIoU_percent": 100.0 * intersection_sum / max(1.0, union_sum),
            "N_acc_percent": 100.0 * empty_correct / max(1, empty_count),
            "positive_count": positive_count,
            "empty_count": empty_count,
        },
        "memory": {
            "peak_allocated_mb": peak_allocated,
            "peak_reserved_mb": peak_reserved,
        },
        "checkpoints": {
            "teacher": os.path.abspath(model_args.teacher_checkpoint),
            "student": os.path.abspath(model_args.student_checkpoint),
        },
    }
    # Keep the two decoder paths side by side instead of letting the head run
    # overwrite the SAM summary it is meant to be compared against.
    tag = str(benchmark_args.decode_path).lower()
    json_path = os.path.join(benchmark_args.output_dir, f"latency_benchmark_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    csv_path = os.path.join(benchmark_args.output_dir, f"latency_samples_{tag}.csv")
    if benchmark_args.save_sample_rows:
        fieldnames = list(rows[0].keys()) if rows else [
            "index", "reference_key", "image_path", "expression",
            "model_latency_ms", "end_to_end_latency_ms", "sample_iou", "target_empty",
        ]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Component latency (mean +/- std ms):")
    for name, stats in summary["latency_ms"]["components"].items():
        print(f"  {name:18s} {stats['mean_ms']:.3f} +/- {stats['std_ms']:.3f}")
    print(f"Saved benchmark summary: {json_path}")
    if benchmark_args.save_sample_rows:
        print(f"Saved per-sample rows: {csv_path}")


if __name__ == "__main__":
    main()





