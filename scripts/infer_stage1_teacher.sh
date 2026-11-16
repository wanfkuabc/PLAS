#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m PLAS.eval.eval_mask_latent_teacher --sam_path "${SAM_PATH:-./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth}" --teacher_checkpoint "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT}" --data_root "${DATA_ROOT:-./PLAS/datasets/refer_seg}" --refer_dataset "${REFER_DATASET:-grefcoco}" --refer_split_by "${REFER_SPLIT_BY:-unc}" --split "${EVAL_SPLIT:-val}" --output_dir "${OUTPUT_DIR:-./PLAS/outputs/stage1_teacher}"

