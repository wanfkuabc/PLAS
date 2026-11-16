#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-./PLAS/pretrained/Qwen3-VL-8B-Instruct}"
SAM_PATH="${SAM_PATH:-./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT}"
DATA_ROOT="${DATA_ROOT:-./PLAS/datasets/refer_seg}"
OUTPUT_DIR="${OUTPUT_DIR:-./PLAS/checkpoints/stage2}"
RUN_PHASE="${RUN_PHASE:-latent_warmup}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-False}"
MASTER_PORT="${MASTER_PORT:-29668}"
QWEN_IMAGE_SIZE="${QWEN_IMAGE_SIZE:-1024}"
LATENT_WIDTH="${LATENT_WIDTH:-512}"
LATENT_DEPTH="${LATENT_DEPTH:-4}"
TEXT_CROSS_ATTENTION_HEADS="${TEXT_CROSS_ATTENTION_HEADS:-8}"
SAM_DECODER_TUNING="${SAM_DECODER_TUNING:-student_b}"
DECODER_ONLY="${DECODER_ONLY:-False}"
USE_FLASH_ATTENTION_2="${USE_FLASH_ATTENTION_2:-True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for p in "$QWEN_MODEL_PATH" "$SAM_PATH" "$TEACHER_CHECKPOINT" "$DATA_ROOT"; do [ -e "$p" ] || { echo "[Error] Missing: $p" >&2; exit 1; }; done
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ -n "$DEEPSPEED_CONFIG" ]; then DS_CONFIG="$DEEPSPEED_CONFIG"; elif [ "${#GPUS[@]}" -eq 1 ]; then DS_CONFIG="./PLAS/configs/stage2_zero2_single_gpu.json"; elif [[ "${OFFLOAD_OPTIMIZER,,}" == "true" || "$OFFLOAD_OPTIMIZER" == "1" ]]; then DS_CONFIG="./PLAS/configs/stage2_zero2_offload.json"; else DS_CONFIG="./PLAS/configs/stage2_zero2.json"; fi
[ -e "$DS_CONFIG" ] || { echo "[Error] Missing: $DS_CONFIG" >&2; exit 1; }
if [ "$RUN_PHASE" = latent_warmup ]; then
  TRAINING_PHASE=latent_warmup; PHASE_DIR="$OUTPUT_DIR/phaseA_latent_warmup"; PHASE_EPOCHS="${PHASE_EPOCHS:-2}"; LATENT_WEIGHT=1.0; HEAD_BCE_WEIGHT=1.0; HEAD_DICE_WEIGHT=1.0; MASK_BCE_WEIGHT=0.0; MASK_DICE_WEIGHT=0.0
elif [ "$RUN_PHASE" = task_adaptation ]; then
  TRAINING_PHASE=task_adaptation; PHASE_DIR="$OUTPUT_DIR/phaseB_task_adaptation"; PHASE_EPOCHS="${PHASE_EPOCHS:-2}"; LATENT_WEIGHT=1.0; HEAD_BCE_WEIGHT=0.0; HEAD_DICE_WEIGHT=0.0; MASK_BCE_WEIGHT=2.0; MASK_DICE_WEIGHT=2.0; [ -n "$STUDENT_CHECKPOINT" ] || { echo "[Error] Phase B requires STUDENT_CHECKPOINT" >&2; exit 1; }
else echo "[Error] RUN_PHASE must be latent_warmup or task_adaptation" >&2; exit 1; fi
extra=(); [ -z "$STUDENT_CHECKPOINT" ] || extra+=(--student_checkpoint "$STUDENT_CHECKPOINT")
resume_args=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  [ -d "$RESUME_FROM_CHECKPOINT" ] || { echo "[Error] Resume directory not found: $RESUME_FROM_CHECKPOINT" >&2; exit 1; }
  [ -f "$RESUME_FROM_CHECKPOINT/qwen_latent_student.pt" ] || { echo "[Error] Resume directory must contain qwen_latent_student.pt" >&2; exit 1; }
  resume_args+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
unset CUDA_VISIBLE_DEVICES
deep=deepspeed
$deep --master_port "$MASTER_PORT" --include "localhost:${GPU_LIST}" PLAS/train/train_qwen_dense_latent_refseg.py --stage student --training_phase "$TRAINING_PHASE" --qwen_model_path "$QWEN_MODEL_PATH" --sam_path "$SAM_PATH" --teacher_checkpoint "$TEACHER_CHECKPOINT" "${extra[@]}" "${resume_args[@]}" --qwen_lora_r 64 --qwen_lora_alpha 128 --qwen_lora_dropout 0.05 --qwen_lora_target_mlp True --readout_layers=-1,-9,-17 --latent_width "$LATENT_WIDTH" --latent_depth "$LATENT_DEPTH" --text_cross_attention_heads "$TEXT_CROSS_ATTENTION_HEADS" --latent_weight "$LATENT_WEIGHT" --relative_latent_weight "${RELATIVE_LATENT_WEIGHT:-0.1}" --head_bce_weight "$HEAD_BCE_WEIGHT" --head_dice_weight "$HEAD_DICE_WEIGHT" --mask_bce_weight "$MASK_BCE_WEIGHT" --mask_dice_weight "$MASK_DICE_WEIGHT" --smooth_rank_weight 0.0 --smooth_rank_margin 0.20 --smooth_rank_temperature 0.05 --phase_a_sam_mask_weight 0.0 --phase_a_sam_bce_weight 0.5 --phase_a_sam_dice_weight 0.5 --decoder_only "$DECODER_ONLY" --sam_decoder_tuning "$SAM_DECODER_TUNING" --hard_negative_cache '' --hard_negative_max 0 --qwen_lora_learning_rate 2e-5 --adapter_learning_rate 1e-4 --calibrator_scale_learning_rate 1e-3 --prompt_adapter_learning_rate 5e-5 --decoder_half_learning_rate 1e-5 --qwen_image_size "$QWEN_IMAGE_SIZE" --use_flash_attention_2 "$USE_FLASH_ATTENTION_2" --data_root "$DATA_ROOT" --refer_dataset refcoco --refer_split_by unc --train_datasets "${TRAIN_DATASETS:-refcoco|refcoco+|refcocog|grefcoco}" --train_split_bys "${TRAIN_SPLIT_BYS:-unc|unc|umd|unc}" --train_split train --dataset_sampling balanced --positive_per_empty 3 --output_dir "$PHASE_DIR" --num_train_epochs "$PHASE_EPOCHS" --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" --gradient_accumulation_steps "${GRAD_ACCUM:-16}" --learning_rate 1e-4 --weight_decay 0.01 --warmup_steps 0 --lr_scheduler_type cosine --bf16 True --gradient_checkpointing True --max_grad_norm 1.0 --logging_steps 10 --save_strategy epoch --save_total_limit 3 --report_to none --dataloader_num_workers "${NUM_WORKERS:-4}" --remove_unused_columns False --dense_mask_log_details True --deepspeed "$DS_CONFIG"
echo "Saved: $PHASE_DIR/qwen_latent_student.pt"





