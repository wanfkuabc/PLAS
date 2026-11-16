#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python PLAS/eval/benchmark_qwen_dense_latent_refseg.py --stage student --training_phase task_adaptation --qwen_model_path "${QWEN_MODEL_PATH:-./PLAS/pretrained/Qwen3-VL-8B-Instruct}" --sam_path "${SAM_PATH:-./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth}" --sam_model_type vit_h --teacher_checkpoint "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT}" --student_checkpoint "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT}" --qwen_lora_r 64 --qwen_lora_alpha 128 --qwen_lora_dropout 0.05 --qwen_lora_target_mlp True --readout_layers=-1,-9,-17 --latent_width 512 --latent_depth 4 --text_cross_attention_heads 8 --sam_decoder_tuning student_b --qwen_image_size "${QWEN_IMAGE_SIZE:-1024}" --use_flash_attention_2 True --data_root "${DATA_ROOT:-./PLAS/datasets/refer_seg}" --refer_dataset "${REFER_DATASET:-refcoco}" --refer_split_by "${REFER_SPLIT_BY:-unc}" --split "${EVAL_SPLIT:-val}" --threshold "${THRESHOLD:-0.55}" --decode_path "${DECODE_PATH:-sam}" --warmup "${WARMUP:-30}" --max_samples "${MAX_SAMPLES:-10}" --exclude_data_loading --output_dir "${OUTPUT_DIR:-./PLAS/outputs/latency}"

