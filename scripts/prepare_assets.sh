#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# One-time import from the old repository layout.  All later commands use PLAS/ only.
mkdir -p PLAS/pretrained/Qwen3-VL-8B-Instruct PLAS/pretrained/sam_vit_h PLAS/datasets/refer_seg PLAS/datasets/ReasonSeg PLAS/checkpoints/stage1 PLAS/checkpoints/stage2/phaseA_latent_warmup PLAS/checkpoints/stage2/phaseB_task_adaptation
[ -d pre_trained/Qwen3-VL-8B-Instruct ] && cp -a pre_trained/Qwen3-VL-8B-Instruct/. PLAS/pretrained/Qwen3-VL-8B-Instruct/ || echo '[Skip] pre_trained/Qwen3-VL-8B-Instruct not found'
[ -f pre_trained/sam_vit_h/sam_vit_h_4b8939.pth ] && cp -f pre_trained/sam_vit_h/sam_vit_h_4b8939.pth PLAS/pretrained/sam_vit_h/ || echo '[Skip] SAM checkpoint not found'
[ -d playground/data/refer_seg ] && cp -a playground/data/refer_seg/. PLAS/datasets/refer_seg/ || echo '[Skip] RefCOCO data not found'
[ -d playground/data/ReasonSeg ] && cp -a playground/data/ReasonSeg/. PLAS/datasets/ReasonSeg/ || echo '[Skip] ReasonSeg data not found'
[ -f checkpoints/llava6-3-2/stage1/full/mask_latent_teacher.pt ] && cp -f checkpoints/llava6-3-2/stage1/full/mask_latent_teacher.pt PLAS/checkpoints/stage1/ || echo '[Skip] Stage-I Teacher not found'
[ -f checkpoints/llava6-3-8/stage2-1024/phaseA_latent_warmup/qwen_latent_student.pt ] && cp -f checkpoints/llava6-3-8/stage2-1024/phaseA_latent_warmup/qwen_latent_student.pt PLAS/checkpoints/stage2/phaseA_latent_warmup/ || true
[ -f checkpoints/llava6-3-8/stage2-1024/phaseB_task_adaptation/qwen_latent_student.pt ] && cp -f checkpoints/llava6-3-8/stage2-1024/phaseB_task_adaptation/qwen_latent_student.pt PLAS/checkpoints/stage2/phaseB_task_adaptation/ || true
echo 'Asset import finished. Check PLAS/pretrained, PLAS/datasets and PLAS/checkpoints.'

