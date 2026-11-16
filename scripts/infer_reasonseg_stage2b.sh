#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-./PLAS/pretrained/Qwen3-VL-8B-Instruct}"
SAM_PATH="${SAM_PATH:-./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT}"
REASONSEG_ROOT="${REASONSEG_ROOT:?Set REASONSEG_ROOT to PLAS/datasets/ReasonSeg}"
OUTPUT_DIR="${OUTPUT_DIR:-./PLAS/outputs/reasonseg_stage2b}"
CUDA_VISIBLE_DEVICES="$GPU" python -m PLAS.eval.eval_qwen_dense_latent_reasonseg --stage student --training_phase task_adaptation --qwen_model_path "$QWEN_MODEL_PATH" --sam_path "$SAM_PATH" --sam_model_type vit_h --teacher_checkpoint "$TEACHER_CHECKPOINT" --student_checkpoint "$STUDENT_CHECKPOINT" --qwen_lora_r 64 --qwen_lora_alpha 128 --qwen_lora_dropout 0.05 --qwen_lora_target_mlp True --readout_layers=-1,-9,-17 --latent_width 512 --latent_depth 4 --text_cross_attention_heads 8 --sam_decoder_tuning student_b --qwen_image_size "${QWEN_IMAGE_SIZE:-1024}" --use_flash_attention_2 True --reasonseg_root "$REASONSEG_ROOT" --split "${EVAL_SPLIT:-val}" --reasonseg_text_index 0 --reasonseg_prompt_mode official --threshold "${THRESHOLD:-0.55}" --save_visuals True --visual_max_samples "${VISUAL_MAX_SAMPLES:-256}" --output_dir "$OUTPUT_DIR"

