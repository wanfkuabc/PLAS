"""Original-mask RefCOCO/G-RefCOCO data path for the llava6-3-2 branch.

Identical decoding to llava6-3-1 with two changes:

* the exclusive-relation partner mask is gone (its loss term was removed);
* empty/positive balancing is a ratio instead of a hard 1:1, because forcing
  50% no-target expressions biased the Student toward predicting nothing.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from PLAS.train.qwen_sam_geometry import sam_image_input, sam_resize_size


GROUP_SINGLE, GROUP_MULTIPLE, GROUP_EMPTY = 0, 1, 2


def _select_refs(refs: List[Dict], split: str) -> List[Dict]:
    # RefCOCO/RefCOCO+ use testA/testB while RefCOCOg-UMD uses test.
    # Substring matching is wrong for split="test": ``'t' in 'train'``
    # includes the entire training split in the reported test set.
    return [ref for ref in refs if ref["split"] == split]


def _decode_annotation(annotation: Dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation", [])
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):
        entries = [dict(segmentation)]
    elif isinstance(segmentation[0], (list, tuple)):
        decoded = mask_utils.decode(mask_utils.frPyObjects(segmentation, height, width))
        return ((decoded.sum(axis=2) if decoded.ndim == 3 else decoded) > 0).astype(np.uint8)
    elif isinstance(segmentation[0], dict):
        entries = [dict(item) for item in segmentation]
    else:
        decoded = mask_utils.decode(mask_utils.frPyObjects([segmentation], height, width))
        return ((decoded.sum(axis=2) if decoded.ndim == 3 else decoded) > 0).astype(np.uint8)
    rles = []
    for entry in entries:
        counts = entry.get("counts")
        if isinstance(counts, (list, tuple)):
            rles.append(mask_utils.frPyObjects(entry, height, width))
        elif isinstance(counts, str):
            entry["counts"] = counts.encode()
            rles.append(entry)
        elif isinstance(counts, bytes):
            rles.append(entry)
        else:
            raise TypeError(f"Unsupported COCO RLE counts: {type(counts)!r}")
    decoded = mask_utils.decode(rles)
    return ((decoded.sum(axis=2) if decoded.ndim == 3 else decoded) > 0).astype(np.uint8)


def _instances_path(data_root: str, dataset: str) -> str:
    primary = os.path.join(data_root, dataset, "instances.json")
    shared = os.path.join(data_root, "refcoco", "instances.json")
    if dataset == "grefcoco" and not os.path.isfile(primary) and os.path.isfile(shared):
        return shared
    return primary


def _image_path(data_root: str, dataset: str, file_name: str) -> str:
    if dataset == "refclef":
        return os.path.join(data_root, "images", "saiapr_tc-12", file_name)
    return os.path.join(data_root, "images", "coco_2014", "train2014", file_name)


class OriginalMaskReferDataset(Dataset):
    """One original-resolution union mask for each referring expression."""

    def __init__(self, data_args):
        allowed = {"refcoco", "refcoco+", "refcocog", "refclef", "grefcoco"}
        if data_args.refer_dataset not in allowed:
            raise ValueError(f"Unsupported referring dataset: {data_args.refer_dataset}")
        dataset_dir = os.path.join(data_args.data_root, data_args.refer_dataset)
        is_gref = data_args.refer_dataset == "grefcoco"
        refs_path = os.path.join(
            dataset_dir,
            f"grefs({data_args.refer_split_by}).json" if is_gref else f"refs({data_args.refer_split_by}).p",
        )
        instances_path = _instances_path(data_args.data_root, data_args.refer_dataset)
        if not os.path.isfile(refs_path) or not os.path.isfile(instances_path):
            raise FileNotFoundError(f"Missing referring annotations: {refs_path} / {instances_path}")
        with open(refs_path, "r" if is_gref else "rb") as handle:
            refs = json.load(handle) if is_gref else pickle.load(handle)
        with open(instances_path, "r", encoding="utf-8") as handle:
            instances = json.load(handle)
        images = {item["id"]: item for item in instances["images"]}
        annotations = {item["id"]: item for item in instances["annotations"]}
        self.records = []
        for reference in _select_refs(refs, data_args.train_split):
            image = images.get(reference["image_id"])
            ann_ids = reference["ann_id"] if isinstance(reference["ann_id"], list) else [reference["ann_id"]]
            annos = [] if ann_ids == [-1] else [annotations.get(int(item)) for item in ann_ids]
            if image is None or any(item is None for item in annos):
                raise RuntimeError(f"Broken reference id={reference.get('ref_id')}")
            group = GROUP_EMPTY if ann_ids == [-1] else (GROUP_MULTIPLE if len(annos) > 1 else GROUP_SINGLE)
            for sentence in reference["sentences"]:
                expression = sentence["sent"].strip().lower()
                if expression:
                    key = f"{reference.get('ref_id', reference['image_id'])}:{','.join(map(str, ann_ids))}:{sentence.get('sent_id', expression)}"
                    self.records.append((
                        image, annos, expression, group, key,
                        int(reference["image_id"]), tuple(map(int, ann_ids)),
                    ))
        ratio = float(data_args.train_subset_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("train_subset_ratio must be in (0, 1].")
        if ratio < 1.0:
            count = max(1, int(len(self.records) * ratio))
            picks = sorted(random.Random(int(data_args.train_subset_seed)).sample(range(len(self.records)), count))
            self.records = [self.records[index] for index in picks]
        # Keep this a separate name: reusing ``ratio`` shadowed train_subset_ratio
        # and made the summary line below report the wrong number.
        synthetic_ratio = float(getattr(data_args, "synthetic_negative_ratio", 0.0))
        real_empty = sum(1 for record in self.records if int(record[3]) == GROUP_EMPTY)
        if synthetic_ratio > 0.0:
            self.records.extend(
                self._synthetic_negatives(instances, synthetic_ratio, int(data_args.train_subset_seed))
            )
        total_empty = sum(1 for record in self.records if int(record[3]) == GROUP_EMPTY)
        self.data_args = data_args
        self._attach_hard_negatives(instances, data_args)
        # Phase A never runs SAM, and a 1024x1024 float32 SAM canvas is ~12 MB
        # per sample.  Skipping it makes the alignment phase I/O-cheap.
        self.load_sam_images = bool(getattr(data_args, "load_sam_images", True))
        print(
            f"llava6-3-8 original-mask set: {len(self.records)} expressions "
            f"({data_args.refer_dataset}|{data_args.refer_split_by}|{data_args.train_split}), "
            f"subset_ratio={ratio}, no-target pool {real_empty} real + "
            f"{total_empty - real_empty} synthetic = {total_empty}."
        )

    @staticmethod
    def _mentioned_categories(expression: str, patterns) -> set:
        return {name for name, pattern in patterns.items() if pattern.search(expression)}

    def _synthetic_negatives(self, instances: Dict, ratio: float, seed: int) -> List:
        """Pair an expression with an image that provably cannot contain it.

        gRefCOCO only supplies ~19k no-target expressions, and the sampler
        repeats each of them several times per epoch, so the no-target head
        memorizes them: training loss hits its floor while validation N-acc
        stalls around 80%.  COCO annotates all 80 categories exhaustively, so
        "this image contains no <category>" is a fact, not a guess, and an
        unlimited supply of verified negatives can be generated from it.
        """
        categories = {int(item["id"]): str(item["name"]).lower() for item in instances.get("categories", [])}
        if not categories:
            print("Synthetic negatives requested but instances.json has no categories; skipping.")
            return []
        # Word boundaries matter here: without them "bus" matches "business".
        patterns = {
            name: re.compile(r"\b" + re.escape(name) + r"e?s?\b")
            for name in set(categories.values())
        }
        present: Dict[int, set] = {}
        for annotation in instances["annotations"]:
            name = categories.get(int(annotation.get("category_id", -1)))
            if name is not None:
                present.setdefault(int(annotation["image_id"]), set()).add(name)

        pool = []
        images = []
        seen_images = set()
        for record in self.records:
            if int(record[3]) == GROUP_EMPTY:
                continue
            mentioned = self._mentioned_categories(record[2], patterns)
            if mentioned:
                pool.append((record[2], mentioned))
            image_id = int(record[5])
            if image_id not in seen_images:
                seen_images.add(image_id)
                images.append((image_id, record[0]))
        if not pool or not images:
            print("Synthetic negatives requested but no expression mentions a COCO category; skipping.")
            return []

        rng = random.Random(seed + 991)
        wanted = int(len(self.records) * ratio)
        synthetic, attempts = [], 0
        while len(synthetic) < wanted and attempts < wanted * 20:
            attempts += 1
            image_id, image_info = images[rng.randrange(len(images))]
            expression, mentioned = pool[rng.randrange(len(pool))]
            if mentioned & present.get(image_id, set()):
                continue
            key = f"synthetic:{image_id}:{len(synthetic)}"
            synthetic.append((image_info, [], expression, GROUP_EMPTY, key, image_id, (-1,)))
        print(
            f"Synthetic verified negatives: {len(synthetic)} (requested {wanted}, "
            f"{len(pool)} category-grounded expressions over {len(images)} images)."
        )
        return synthetic

    HARD_NEGATIVE_INDEX = "index.json"
    HARD_NEGATIVE_LATENTS = "latents.npy"

    def _attach_hard_negatives(self, instances, data_args) -> None:
        """Attach precomputed same-image/same-category Teacher latents."""
        self.hard_negative_max = int(getattr(data_args, "hard_negative_max", 0) or 0)
        directory = str(getattr(data_args, "hard_negative_cache", "") or "")
        self.hard_negative_ids = None
        self.hard_negative_rows = None
        self.hard_negative_store = None
        if self.hard_negative_max <= 0 or not directory:
            return
        index_path = os.path.join(directory, self.HARD_NEGATIVE_INDEX)
        latent_path = os.path.join(directory, self.HARD_NEGATIVE_LATENTS)
        if not os.path.isfile(index_path) or not os.path.isfile(latent_path):
            raise FileNotFoundError(
                f"hard_negative_max={self.hard_negative_max} but no cache at {directory}. "
                "Build it with PLAS/tools/build_hard_negative_cache.py."
            )
        with open(index_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.hard_negative_rows = {int(key): int(value) for key, value in manifest["rows"].items()}
        self.hard_negative_shape = tuple(int(value) for value in manifest["latent_shape"])
        self.hard_negative_store = np.load(latent_path, mmap_mode="r")
        if tuple(self.hard_negative_store.shape[1:]) != self.hard_negative_shape:
            raise RuntimeError(
                f"Cache latent shape {tuple(self.hard_negative_store.shape[1:])} "
                f"!= manifest {self.hard_negative_shape}"
            )

        by_image = {}
        for annotation in instances["annotations"]:
            by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.hard_negative_ids = []
        covered = 0
        for record in self.records:
            _, annotations, _, _, _, image_id, ann_ids = record
            target_ids = {int(item) for item in ann_ids}
            categories = {int(item.get("category_id", -1)) for item in annotations if item is not None}
            rivals = []
            for annotation in by_image.get(int(image_id), []):
                identifier = int(annotation["id"])
                if identifier in target_ids or int(annotation.get("category_id", -1)) not in categories:
                    continue
                if identifier in self.hard_negative_rows:
                    rivals.append(identifier)
                if len(rivals) >= self.hard_negative_max:
                    break
            self.hard_negative_ids.append(rivals)
            covered += bool(rivals)
        print(
            f"llava6-3-8 smooth-rank cache: {covered}/{len(self.records)} expressions "
            f"({100.0 * covered / max(1, len(self.records)):.1f}%) have a same-category rival."
        )

    def _hard_negative_tensors(self, index: int):
        latents = torch.zeros((self.hard_negative_max, *self.hard_negative_shape), dtype=torch.float32)
        valid = torch.zeros(self.hard_negative_max, dtype=torch.bool)
        for position, identifier in enumerate(self.hard_negative_ids[index][: self.hard_negative_max]):
            latents[position] = torch.from_numpy(
                np.asarray(self.hard_negative_store[self.hard_negative_rows[identifier]], dtype=np.float32)
            )
            valid[position] = True
        return latents, valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_info, annotations, expression, target_group, reference_key, _, _ = self.records[index]
        path = _image_path(self.data_args.data_root, self.data_args.refer_dataset, image_info["file_name"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing referenced image: {path}")
        image = Image.open(path).convert("RGB")
        if image.size != (int(image_info["width"]), int(image_info["height"])):
            raise RuntimeError(f"Image/annotation size mismatch: {path}")
        target = np.zeros((image.height, image.width), dtype=np.uint8)
        for annotation in annotations:
            target |= _decode_annotation(annotation, image.height, image.width)
        sam_canvas = sam_image_input(image) if self.load_sam_images else torch.zeros(3, 1, 1)
        hard_negatives = {}
        if self.hard_negative_ids is not None:
            latents, valid = self._hard_negative_tensors(index)
            hard_negatives = {"distractor_latents": latents, "distractor_valid": valid}
        return {
            **hard_negatives,
            "sam_images": sam_canvas,
            "gt_masks": torch.from_numpy(target[None].astype(np.float32)),
            # Geometry is inference metadata. Do not recover it from a GT mask.
            "original_sizes": torch.tensor([image.height, image.width], dtype=torch.long),
            "sam_input_sizes": torch.tensor(sam_resize_size(image), dtype=torch.long),
            "target_groups": torch.tensor(target_group, dtype=torch.long),
            "reference_key": reference_key,
            "expression": expression,
            "image_path": path,
        }





