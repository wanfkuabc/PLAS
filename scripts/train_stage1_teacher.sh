#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SAM_PATH="${SAM_PATH:-./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./PLAS/datasets/refer_seg}"
OUTPUT_DIR="${OUTPUT_DIR:-./PLAS/checkpoints/stage1}"
DS_CONFIG="${DEEPSPEED_CONFIG:-./PLAS/configs/stage1_zero2.json}"
RUN_STAGE="${RUN_STAGE:-pipeline}"
for p in "$SAM_PATH" "$DATA_ROOT" "$DS_CONFIG"; do [ -e "$p" ] || { echo "[Error] Missing: $p" >&2; exit 1; }; done
WARMUP_DIR="$OUTPUT_DIR/warmup"
FULL_DIR="$OUTPUT_DIR/full"
common=(--sam_path "$SAM_PATH" --sam_model_type vit_h --noise_sigma_start 0.0 --noise_sigma_end "${NOISE_SIGMA_END:-0.40}" --noise_warmup_ratio "${NOISE_WARMUP_RATIO:-0.30}" --head_clean_weight "${HEAD_CLEAN_WEIGHT:-0.5}" --teacher_learning_rate "${TEACHER_LR:-2e-5}" --decoder_learning_rate "${DECODER_LR:-1e-6}" --data_root "$DATA_ROOT" --refer_dataset "${REFER_DATASET:-grefcoco}" --refer_split_by "${REFER_SPLIT_BY:-unc}" --train_datasets "${TRAIN_DATASETS:-}" --train_split_bys "${TRAIN_SPLIT_BYS:-}" --train_split train --train_subset_ratio "${TRAIN_SUBSET_RATIO:-1.0}" --deepspeed "$DS_CONFIG" --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-4}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" --learning_rate "${TEACHER_LR:-2e-5}" --lr_scheduler_type cosine --warmup_ratio 0.03 --max_grad_norm 0.5 --adam_epsilon 1e-6 --bf16 False --tf32 True --logging_steps 20 --teacher_log_details True --save_strategy epoch --save_total_limit 1 --eval_strategy no --dataloader_num_workers "${NUM_WORKERS:-4}" --report_to none)
launch(){ unset CUDA_VISIBLE_DEVICES; deepspeed --include "localhost:${GPU_LIST}" "$@"; }
if [ "$RUN_STAGE" = warmup ] || [ "$RUN_STAGE" = pipeline ]; then
  launch PLAS/train/train_mask_latent_teacher.py --tuning_mode warmup "${common[@]}" --output_dir "$WARMUP_DIR" --num_train_epochs "${WARMUP_EPOCHS:-1}"
fi
if [ "$RUN_STAGE" = full ] || [ "$RUN_STAGE" = pipeline ]; then
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$WARMUP_DIR/mask_latent_teacher.pt}"
  [ -f "$RESUME_CHECKPOINT" ] || { echo "[Error] Missing warmup checkpoint: $RESUME_CHECKPOINT" >&2; exit 1; }
  launch PLAS/train/train_mask_latent_teacher.py --tuning_mode full --teacher_checkpoint "$RESUME_CHECKPOINT" "${common[@]}" --output_dir "$FULL_DIR" --num_train_epochs "${FULL_EPOCHS:-2}"
fi
echo "Stage-I Teacher: $FULL_DIR/mask_latent_teacher.pt"

