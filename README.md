<div align="center">

<h1>PLAS: Parallel Layer-Aggregated Spatial Readout with Progressive Latent-Space Alignment for Referring Image Segmentation</h1>

<p><strong>YaoYang Liu, Haixia Cao, Yinglin Chen, Weijun Sun, Xiaozhao Fang, Shengli Xie, Jie Wen</strong></p>

<p>Guangdong University of Technology<br />
Harbin Institute of Technology, Shenzhen</p>

</div>

### Abstract

Referring image segmentation (RIS) aims to ground natural-language expressions in images and produce pixel-accurate masks. Despite advances in multimodal large language models (MLLMs), two challenges remain: efficiently extracting spatial information from their hidden states and constraining the resulting representations to preserve target-mask geometry. To address this, we propose **PLAS** (Parallel Layer-Aggregated Spatial Readout with Progressive Latent-Space Alignment), an efficient RIS framework coupling parallel spatial readout with geometry-supervised latent learning. Specifically, PLAS introduces learnable spatial readout tokens into the MLLM and extracts their hidden states at multiple Transformer depths within a single MLLM forward pass without autoregressive segmentation-token generation, aggregating them with learnable layer-wise weights to produce spatially organized multimodal representations. Furthermore, we develop a progressive teacher--student framework that learns decodable mask latents from annotated masks and transfers their geometric structure to student representations through explicit latent-space alignment, followed by image-conditioned adaptation. The predicted latents are then converted into dense prompts for SAM decoding. Extensive experiments across five challenging benchmarks demonstrate the competitive and leading performance of PLAS. On a single NVIDIA RTX 4090D, PLAS achieves a model-side inference latency of **0.43 s per image**, making it **1.63x** faster than the fastest competing method and **2.49x** faster than a method with comparable segmentation accuracy.

### Three-stage training and inference pipeline

<div align="center">
<img src="./author-kit-main/fig/plas_three_stage_pipeline.png" alt="PLAS three-stage training and inference pipeline" width="920" />
</div>

<p align="center"><em>Overview of PLAS and its reported accuracy--efficiency measurements. Existing MLLM-based segmentation methods employ token embeddings, localization symbols, or patch-level predictions to connect multimodal representations with mask prediction. In contrast, PLAS introduces learnable spatial readout tokens within the MLLM, whose hidden states are jointly extracted in a single forward pass under causal attention, without autoregressive segmentation-token generation. The extracted features are subsequently refined through bidirectional spatial interactions, mapped into mask latents, explicitly aligned with mask-level geometry during training, and transformed into dense prompts for image-conditioned SAM decoding. The inset compares segmentation accuracy (cIoU) against model-side inference latency across representative methods.</em></p>

The source figures are available as [PNG](./author-kit-main/fig/plas_three_stage_pipeline.png), [architecture PDF](./author-kit-main/fig/architecture.pdf), and [overview PDF](./author-kit-main/fig/overview.pdf).

## 1. Standalone layout

The original `llava6-3-8/` directory is preserved. The standalone PLAS implementation is contained in this directory and does not import code from other repository branches at runtime.

```text
PLAS/
├── model/
│   ├── qwen_dense_latent_refseg.py
│   ├── mask_latent_teacher.py
│   └── segment_anything/          # Local SAM implementation
├── train/
│   ├── train_mask_latent_teacher.py
│   ├── train_qwen_dense_latent_refseg.py
│   ├── qwen_refseg_data.py
│   └── qwen_sam_geometry.py
├── eval/
│   ├── eval_mask_latent_teacher.py
│   ├── eval_qwen_dense_latent_refseg.py
│   ├── eval_qwen_dense_latent_reasonseg.py
│   └── benchmark_qwen_dense_latent_refseg.py
├── tools/                         # Optional hard-negative utility
├── pretrained/                    # Qwen3-VL and SAM weights
├── datasets/                      # RefCOCO and ReasonSeg data
├── checkpoints/                   # Three-stage checkpoints
├── outputs/                       # Evaluation and benchmark outputs
├── configs/                       # Local DeepSpeed configurations
└── scripts/                       # Training and inference launchers
```

## 2. Environment

The supplied environment file contains the required PyTorch, Transformers, PEFT, DeepSpeed, Qwen-VL utilities, COCO tools, and image-processing packages.

```bash
cd /data2/lyy/Text4Seg-main
conda env create -f PLAS/environment-plas.yml
conda activate text4seg-plas
```

If the environment already exists, activate that environment instead. FlashAttention can be installed separately when the CUDA and compiler versions match:

```bash
pip install -U flash-attn --no-build-isolation
```

## 3. Pretrained model downloads

### Qwen3-VL-8B-Instruct

Official model page: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

```bash
mkdir -p PLAS/pretrained/Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir PLAS/pretrained/Qwen3-VL-8B-Instruct
```

### SAM ViT-H

Official checkpoint: [SAM ViT-H checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

```bash
mkdir -p PLAS/pretrained/sam_vit_h
wget -O PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

The expected local paths are:

```text
PLAS/pretrained/Qwen3-VL-8B-Instruct/
PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth
```

## 4. Dataset downloads and placement

### 4.1 COCO 2014 images and annotations

PLAS uses COCO 2014 train images for RefCOCO, RefCOCO+, RefCOCOg, and gRefCOCO.

- [COCO 2014 train images](http://images.cocodataset.org/zips/train2014.zip)
- [COCO 2014 train/val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- [COCO official download page](https://cocodataset.org/#download)

```bash
mkdir -p PLAS/datasets/refer_seg/images/coco_2014
wget -O /tmp/train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip /tmp/train2014.zip -d PLAS/datasets/refer_seg/images/coco_2014
```

The resulting image directory must be:

```text
PLAS/datasets/refer_seg/images/coco_2014/train2014/
```

### 4.2 RefCOCO, RefCOCO+, and RefCOCOg

Official API and cleaned annotation repository: [lichengunc/refer](https://github.com/lichengunc/refer)

Direct annotation archives:

- [RefCOCO](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
- [RefCOCO+](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
- [RefCOCOg](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)

If the original UNC server is unavailable, use the alternative mirrors listed in the [refer repository download instructions](https://github.com/lichengunc/refer#download).

Extract the archives so that the annotation files are available as follows:

```text
PLAS/datasets/refer_seg/
├── refcoco/
│   ├── refs(unc).p
│   └── instances.json
├── refcoco+/
│   ├── refs(unc).p
│   └── instances.json
└── refcocog/
    ├── refs(umd).p
    └── instances.json
```

The default training recipe uses `refcoco|refcoco+|refcocog|grefcoco` with split identifiers `unc|unc|umd|unc`.

### 4.3 gRefCOCO

Official repository and data instructions: [henghuiding/gRefCOCO](https://github.com/henghuiding/gRefCOCO)

Place the generalized referring-expression annotations under:

```text
PLAS/datasets/refer_seg/grefcoco/
└── grefs(unc).json
```

The gRefCOCO loader can reuse `PLAS/datasets/refer_seg/refcoco/instances.json` when a separate `grefcoco/instances.json` is not provided.

### 4.4 ReasonSeg

ReasonSeg is released with the LISA project.

- [LISA repository and dataset instructions](https://github.com/JIA-Lab-research/LISA)
- [ReasonSeg Google Drive folder](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing)

The PLAS ReasonSeg evaluator expects each split directory to contain paired image and LabelMe-style JSON files:

```text
PLAS/datasets/ReasonSeg/
├── val/
│   ├── sample_000.jpg
│   ├── sample_000.json
│   └── ...
└── test/
    ├── sample_000.jpg
    ├── sample_000.json
    └── ...
```

The evaluator also accepts a root directory containing `ReasonSeg_val/` or `ReasonSeg_test/` directories. Set `REASONSEG_ROOT` to the directory containing the split folders.

### 4.5 One-time asset import from the old repository layout

If the old repository assets are already available, this script copies them into the standalone PLAS directories:

```bash
cd /data2/lyy/Text4Seg-main
bash PLAS/scripts/prepare_assets.sh
```

After this import, training and inference commands use only `PLAS/pretrained`, `PLAS/datasets`, and `PLAS/checkpoints`.

## 5. Checkpoint layout

```text
PLAS/checkpoints/
├── stage1/
│   └── full/mask_latent_teacher.pt
└── stage2/
    ├── phaseA_latent_warmup/qwen_latent_student.pt
    └── phaseB_task_adaptation/qwen_latent_student.pt
```

To import existing checkpoints from the old layout, run `PLAS/scripts/prepare_assets.sh` or copy them manually into the paths above.

## 6. Three-stage training

### Stage I: Teacher pretraining

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_STAGE=pipeline \
SAM_PATH=./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
DATA_ROOT=./PLAS/datasets/refer_seg \
OUTPUT_DIR=./PLAS/checkpoints/stage1 \
bash PLAS/scripts/train_stage1_teacher.sh
```

Outputs:

```text
PLAS/checkpoints/stage1/warmup/mask_latent_teacher.pt
PLAS/checkpoints/stage1/full/mask_latent_teacher.pt
```

### Stage II-A: Latent warmup

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
RUN_PHASE=latent_warmup \
QWEN_MODEL_PATH=./PLAS/pretrained/Qwen3-VL-8B-Instruct \
SAM_PATH=./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
DATA_ROOT=./PLAS/datasets/refer_seg \
OUTPUT_DIR=./PLAS/checkpoints/stage2 \
bash PLAS/scripts/train_stage2.sh
```

Stage II-A optimizes latent alignment together with image-free BCE and Dice reconstruction losses.

### Stage II-B: Image-conditioned task adaptation

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseA_latent_warmup/qwen_latent_student.pt \
RUN_PHASE=task_adaptation \
QWEN_MODEL_PATH=./PLAS/pretrained/Qwen3-VL-8B-Instruct \
SAM_PATH=./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
DATA_ROOT=./PLAS/datasets/refer_seg \
OUTPUT_DIR=./PLAS/checkpoints/stage2 \
bash PLAS/scripts/train_stage2.sh
```

Output:

```text
PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt
```

## 7. Three-stage inference and evaluation

### Stage I: Teacher reconstruction evaluation

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
SAM_PATH=./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
DATA_ROOT=./PLAS/datasets/refer_seg \
OUTPUT_DIR=./PLAS/outputs/stage1_teacher \
bash PLAS/scripts/infer_stage1_teacher.sh
```

### Stage II-A: RefCOCO evaluation

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseA_latent_warmup/qwen_latent_student.pt \
EVAL_SPLIT=val \
OUTPUT_DIR=./PLAS/outputs/refcoco_stage2a \
bash PLAS/scripts/infer_refseg_stage2a.sh
```

### Stage II-B: RefCOCO evaluation

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt \
EVAL_SPLIT=val \
OUTPUT_DIR=./PLAS/outputs/refcoco_stage2b \
bash PLAS/scripts/infer_refseg_stage2b.sh
```

### Stage II-B: ReasonSeg evaluation

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt \
REASONSEG_ROOT=./PLAS/datasets/ReasonSeg \
EVAL_SPLIT=val \
OUTPUT_DIR=./PLAS/outputs/reasonseg_stage2b_val \
bash PLAS/scripts/infer_reasonseg_stage2b.sh
```

Use `EVAL_SPLIT=test` for the ReasonSeg test split.

## 8. Latency benchmark

The latency benchmark is intentionally single-process and single-GPU. Do not launch it with `torchrun` or `deepspeed`.

Smoke test:

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt \
MAX_SAMPLES=10 \
WARMUP=30 \
OUTPUT_DIR=./PLAS/outputs/latency_10 \
bash PLAS/scripts/benchmark_latency.sh
```

Set `MAX_SAMPLES=0` for the complete validation split. The benchmark writes `latency_benchmark_sam.json` and `latency_samples_sam.csv`.

## 9. Verification

Run these checks from the repository root on the server:

```bash
cd /data2/lyy/Text4Seg-main
for script in PLAS/scripts/*.sh; do bash -n "$script"; done
python -m compileall -q PLAS
python -c "from PLAS.model.segment_anything import sam_model_registry; print(sorted(sam_model_registry))"
```

The standalone implementation uses local `PLAS.*` imports. The Qwen3 model name is retained only where it refers to the actual pretrained Qwen3-VL checkpoint.

