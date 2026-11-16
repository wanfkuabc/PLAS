"""llava6-3-4 Stage-1 training: the regressable mask latent.

Run it in two passes, exactly like the earlier Teacher recipe:

    warmup : mask encoder + mask head + prompt adapter, SAM decoder frozen
    full   : the same modules plus SAM's mask decoder at a very low LR

The noise curriculum is the point of this stage.  ``--noise_sigma_end`` is the
relative latent error the decoding path must tolerate; set it slightly above
the Student's measured ``latent_error`` (the diagnostic script reports it).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset

from PLAS.model.mask_latent_teacher import MaskLatentTeacher
from PLAS.train.qwen_refseg_data import OriginalMaskReferDataset


@dataclass
class ModelArguments:
    sam_path: str = "./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    teacher_checkpoint: str = ""
    tuning_mode: str = "warmup"
    noise_sigma_start: float = 0.0
    noise_sigma_end: float = 0.40
    noise_warmup_ratio: float = 0.30
    sam_bce_weight: float = 1.0
    sam_dice_weight: float = 1.0
    head_bce_weight: float = 1.0
    head_dice_weight: float = 1.0
    head_clean_weight: float = 0.5
    teacher_learning_rate: float = 2e-5
    decoder_learning_rate: float = 1e-6


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


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    teacher_log_details: bool = field(default=True)
    teacher_abort_loss: float = field(default=100.0)


def _parse_sources(value: str):
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _default_split_by(dataset: str) -> str:
    return "umd" if dataset == "refcocog" else "unc"


def build_dataset(data_args: DataArguments) -> Dataset:
    names = _parse_sources(data_args.train_datasets)
    if not names:
        return OriginalMaskReferDataset(data_args)
    split_bys = _parse_sources(data_args.train_split_bys) or [_default_split_by(name) for name in names]
    if len(split_bys) != len(names):
        raise ValueError("--train_split_bys must be empty or match --train_datasets in length.")
    children = [
        OriginalMaskReferDataset(replace(data_args, refer_dataset=name, refer_split_by=split_by, train_datasets="", train_split_bys=""))
        for name, split_by in zip(names, split_bys)
    ]
    print("Stage-1 mixture:", ", ".join(f"{name}={len(child)}" for name, child in zip(names, children)))
    return ConcatDataset(children)


class TeacherCollator:
    def __call__(self, instances):
        return {
            "sam_images": [item["sam_images"] for item in instances],
            "gt_masks": [item["gt_masks"] for item in instances],
            "sam_input_sizes": torch.stack([item["sam_input_sizes"] for item in instances]),
            "original_sizes": torch.stack([item["original_sizes"] for item in instances]),
            "target_groups": torch.stack([item["target_groups"] for item in instances]),
        }


class TeacherTrainer(transformers.Trainer):
    def __init__(self, *args, model_args: ModelArguments, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self._detail_sums = {}
        self._detail_count = 0

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_teacher(os.path.join(output_dir, "mask_latent_teacher.pt"))
        with open(os.path.join(output_dir, "mask_latent_teacher_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"format": model.FORMAT, "model": asdict(self.model_args)}, handle, indent=2)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        raw = self.model.module if hasattr(self.model, "module") else self.model
        encoder_names = raw.encoder_head_parameter_names()
        decoder_names = raw.sam_decoder_parameter_names()
        encoder, decoder = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            normalized = name.removeprefix("module.")
            if normalized in encoder_names:
                encoder.append(parameter)
            elif normalized in decoder_names:
                decoder.append(parameter)
            else:
                raise RuntimeError(f"Unexpected trainable Teacher parameter: {normalized}")
        groups = [
            {"params": encoder, "lr": float(self.model_args.teacher_learning_rate), "weight_decay": self.args.weight_decay},
            {"params": decoder, "lr": float(self.model_args.decoder_learning_rate), "weight_decay": self.args.weight_decay},
        ]
        groups = [group for group in groups if group["params"]]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def _current_sigma(self) -> float:
        start = float(self.model_args.noise_sigma_start)
        end = float(self.model_args.noise_sigma_end)
        warmup = min(0.95, max(0.0, float(self.model_args.noise_warmup_ratio)))
        progress = float(self.state.global_step) / max(1, int(self.state.max_steps))
        if progress <= warmup:
            return start
        span = (progress - warmup) / max(1e-6, 1.0 - warmup)
        return start + (end - start) * min(1.0, span)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raw = model.module if hasattr(model, "module") else model
        raw.set_noise_sigma(self._current_sigma())
        output = model(**inputs)
        loss = output["loss"]
        if not torch.isfinite(loss).all() or float(loss.detach().float()) > float(self.args.teacher_abort_loss):
            terms = {key: float(value.detach().float()) for key, value in output.items() if key.startswith("loss_")}
            raise FloatingPointError(f"Stage-1 loss diverged: {float(loss.detach().float())}; {terms}")
        if model.training and bool(self.args.teacher_log_details):
            values = {key: float(value.detach().float().cpu()) for key, value in output.items() if key != "loss"}
            values["loss_total"] = float(loss.detach().float().cpu())
            for name, value in values.items():
                self._detail_sums[name] = self._detail_sums.get(name, 0.0) + value
            self._detail_count += 1
            at_boundary = bool(getattr(self.accelerator, "sync_gradients", True))
            due = (int(self.state.global_step) + 1) % max(1, int(self.args.logging_steps)) == 0
            if at_boundary and due:
                if self.is_world_process_zero() and self._detail_count:
                    self.log({f"teacher/{name}": value / self._detail_count for name, value in self._detail_sums.items()})
                self._detail_sums.clear()
                self._detail_count = 0
        return (loss, output) if return_outputs else loss


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.tuning_mode not in {"warmup", "full"}:
        raise ValueError("--tuning_mode must be warmup or full.")
    if not os.path.isfile(model_args.sam_path):
        raise FileNotFoundError(f"Missing SAM checkpoint: {model_args.sam_path}")
    model = MaskLatentTeacher(model_args.sam_path, model_args.sam_model_type)
    if model_args.teacher_checkpoint:
        if not os.path.isfile(model_args.teacher_checkpoint):
            raise FileNotFoundError(f"Missing Teacher checkpoint: {model_args.teacher_checkpoint}")
        model.load_teacher(model_args.teacher_checkpoint)
        print(f"Resumed Stage-1 weights from {model_args.teacher_checkpoint}; optimizer restarts fresh.")
    model.configure_tuning(model_args.tuning_mode)
    model.sam_bce_weight = float(model_args.sam_bce_weight)
    model.sam_dice_weight = float(model_args.sam_dice_weight)
    model.head_bce_weight = float(model_args.head_bce_weight)
    model.head_dice_weight = float(model_args.head_dice_weight)
    model.head_clean_weight = float(model_args.head_clean_weight)
    model.install_trainable_gradient_guards()
    dataset = build_dataset(data_args)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage-1 mode={model_args.tuning_mode}; trainable parameters={trainable:,}")
    trainer = TeacherTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=TeacherCollator(), model_args=model_args,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint or None)
    if trainer.is_world_process_zero():
        trainer._save(training_args.output_dir)


if __name__ == "__main__":
    main()





