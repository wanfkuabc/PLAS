"""Precompute Teacher latents for same-category rival instances.

The hard-negative margin needs d(Z_pred, Z_distractor) every step.  Encoding
those masks online would run the Teacher mask encoder K extra times per sample
for tensors that never change: the Teacher is frozen, and a latent depends only
on (annotation, image geometry), both fixed.  So they are computed once here.

Layout, written to ``--cache_dir``:

    latents.npy   float16 [N, 32, 32, 32], one row per annotation
    index.json    {"rows": {ann_id: row}, "latent_shape": [...], ...}

``latents.npy`` is read back with ``mmap_mode="r"``, so every rank shares one
copy through the page cache instead of holding its own.  At float16 a row is
64 KB; a RefCOCO/+/g train build is typically 4-8 GB on disk and costs almost
no resident memory.

Only annotations that some training expression can actually use are stored:
same image, same category, not the target itself.

    bash scripts/v1_5/build_hard_negative_cache_PLAS.sh
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from PIL import Image

from PLAS.model.mask_latent_teacher import LATENT_CHANNELS, LATENT_GRID, MaskLatentTeacher
from PLAS.train.qwen_refseg_data import (
    OriginalMaskReferDataset, _decode_annotation, _image_path, _instances_path,
)
from PLAS.train.qwen_sam_geometry import sam_resize_size
from PLAS.train.train_qwen_dense_latent_refseg import DataArguments, ModelArguments


@dataclass
class CacheArguments:
    cache_dir: str = "./PLAS/checkpoints/hard_negative_cache"
    # Must match the training run: a latent is only reusable if the geometry it
    # was built from is the same, and that is fixed per image.
    batch_size: int = 16
    overwrite: bool = False
    local_rank: int = -1


def collect_targets(data_args, instances):
    """Annotations some expression will ask for: same image, same category."""
    by_image = {}
    for annotation in instances["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    images = {int(item["id"]): item for item in instances["images"]}

    dataset = OriginalMaskReferDataset(data_args)
    wanted, per_image = {}, {}
    for record in dataset.records:
        _, annotations, _, _, _, image_id, ann_ids = record
        target_ann_ids = {int(item) for item in ann_ids}
        categories = {int(item.get("category_id", -1)) for item in annotations if item is not None}
        for annotation in by_image.get(int(image_id), []):
            identifier = int(annotation["id"])
            if identifier in target_ann_ids:
                continue
            if int(annotation.get("category_id", -1)) not in categories:
                continue
            if identifier not in wanted:
                wanted[identifier] = annotation
                per_image.setdefault(int(image_id), []).append(identifier)
    return wanted, per_image, images, len(dataset.records)


@torch.no_grad()
def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CacheArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    if not os.path.isfile(model_args.teacher_checkpoint):
        raise FileNotFoundError("Needs --teacher_checkpoint: the cache is Teacher latents.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_path = os.path.join(args.cache_dir, "latents.npy")
    index_path = os.path.join(args.cache_dir, "index.json")
    if os.path.isfile(latent_path) and not bool(args.overwrite):
        raise FileExistsError(f"{latent_path} exists. Pass --overwrite True to rebuild it.")

    # The dataset build must not pull in the cache it is about to create.
    build_args = DataArguments(**{**data_args.__dict__, "hard_negative_cache": "", "hard_negative_max": 0})
    with open(_instances_path(build_args.data_root, build_args.refer_dataset), "r", encoding="utf-8") as handle:
        instances = json.load(handle)
    wanted, per_image, images, expressions = collect_targets(build_args, instances)
    print(f"{expressions} expressions -> {len(wanted)} distinct same-category rival annotations "
          f"across {len(per_image)} images.")
    if not wanted:
        raise RuntimeError("No same-category rivals found; the hard-negative term would never fire.")

    teacher = MaskLatentTeacher(model_args.sam_path, model_args.sam_model_type)
    teacher.load_teacher(model_args.teacher_checkpoint)
    teacher.to(device=device).eval()

    shape = (int(LATENT_CHANNELS), int(LATENT_GRID), int(LATENT_GRID))
    os.makedirs(args.cache_dir, exist_ok=True)
    store = np.lib.format.open_memmap(
        latent_path, mode="w+", dtype=np.float16, shape=(len(wanted), *shape),
    )
    rows = {}
    written = 0
    for position, (image_id, identifiers) in enumerate(per_image.items()):
        info = images.get(int(image_id))
        if info is None:
            raise RuntimeError(f"instances.json has annotations for missing image {image_id}")
        path = _image_path(build_args.data_root, build_args.refer_dataset, info["file_name"])
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if image.size != (int(info["width"]), int(info["height"])):
            raise RuntimeError(f"Image/annotation size mismatch: {path}")
        # Identical geometry to training: the Teacher canvas is derived from the
        # image alone, never from a mask.
        input_size = torch.tensor(sam_resize_size(image), dtype=torch.long, device=device)

        for start in range(0, len(identifiers), int(args.batch_size)):
            chunk = identifiers[start:start + int(args.batch_size)]
            masks, keep = [], []
            for identifier in chunk:
                decoded = _decode_annotation(wanted[identifier], image.height, image.width)
                if not decoded.any():
                    continue
                masks.append(torch.from_numpy(decoded[None].astype(np.float32)))
                keep.append(identifier)
            if not masks:
                continue
            sizes = input_size.unsqueeze(0).expand(len(masks), -1)
            latents, _ = teacher.encode_gt(masks, sizes)
            latents = latents.detach().float().cpu().numpy().astype(np.float16)
            for offset, identifier in enumerate(keep):
                store[written] = latents[offset]
                rows[identifier] = written
                written += 1
        if (position + 1) % 500 == 0:
            print(f"  {position + 1}/{len(per_image)} images, {written} latents", flush=True)

    store.flush()
    del store
    # Trim: annotations with an empty segmentation were skipped.
    if written < len(wanted):
        full = np.load(latent_path, mmap_mode="r")
        trimmed = np.array(full[:written])
        del full
        np.save(latent_path, trimmed)
        print(f"Trimmed {len(wanted) - written} empty-segmentation rows.")

    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({
            "rows": {str(key): int(value) for key, value in rows.items()},
            "latent_shape": list(shape),
            "dataset": build_args.refer_dataset,
            "split_by": build_args.refer_split_by,
            "split": build_args.train_split,
            "train_datasets": build_args.train_datasets,
            "teacher_checkpoint": os.path.abspath(model_args.teacher_checkpoint),
            "count": written,
        }, handle, indent=2)

    size = os.path.getsize(latent_path) / (1024 ** 3)
    print(f"\nWrote {written} latents ({size:.2f} GB) to {args.cache_dir}")
    print("Training reads this through mmap, so all ranks share one copy.")
    print(f"Now pass HARD_NEGATIVE_CACHE={args.cache_dir} to the Phase B run.")


if __name__ == "__main__":
    main()




