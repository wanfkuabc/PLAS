# PLAS Standalone

这是从原 `llava6-3-8` 分支整理出的独立 Qwen3 dense-latent 版本。`PLAS/` 内包含运行这条实验链所需的模型、SAM、数据处理、训练和评估代码；运行时不再导入仓库中的 `llava1`、`llava4`、`llava5`、`llava6-3-8` 或其他分支。

原始 `llava6-3-8/` 目录没有删除。`PLAS/` 是一个独立副本，后续实验只需要使用本目录中的路径。

## 目录结构

```text
PLAS/
├── model/
│   ├── qwen_dense_latent_refseg.py
│   ├── mask_latent_teacher.py
│   └── segment_anything/          # 本地 SAM 实现
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
├── pretrained/                    # 需要放 Qwen3 和 SAM 权重
├── datasets/                      # 需要放数据集
├── checkpoints/                   # 三阶段训练输出
├── outputs/                       # 推理和测速结果
├── configs/                       # 本地 DeepSpeed 配置
└── scripts/                       # 三阶段训练/推理启动脚本
```

## 1. 环境

在服务器上：

```bash
cd /data2/lyy/Text4Seg-main
conda env create -f PLAS/environment-PLAS-vl.yml
conda activate text4seg-plas
```

如果环境已经存在，直接激活原来的 `PLASvl` 环境即可。除 Python 包外，代码不依赖其他仓库分支。

## 2. 预训练模型放置位置

把模型放到下面的位置。只在第一次整理时从旧目录复制，之后所有命令都使用 `PLAS/` 内的路径。

```bash
cd /data2/lyy/Text4Seg-main

mkdir -p PLAS/pretrained/Qwen3-VL-8B-Instruct
mkdir -p PLAS/pretrained/sam_vit_h

cp -a pre_trained/Qwen3-VL-8B-Instruct/. \
  PLAS/pretrained/Qwen3-VL-8B-Instruct/
cp pre_trained/sam_vit_h/sam_vit_h_4b8939.pth \
  PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth
```

整理完成后，训练和推理不再使用 `pre_trained/`。也可以直接执行一次 `bash PLAS/scripts/prepare_assets.sh` 自动复制旧目录中的可用模型、数据和 checkpoint。

## 3. 数据集放置位置

### RefCOCO 系列

目录应为：

```text
PLAS/datasets/refer_seg/
├── images/
│   └── coco_2014/train2014/        # COCO 2014 图片
├── refcoco/
│   ├── refs(unc).p
│   └── instances.json
├── refcoco+/
│   ├── refs(unc).p
│   └── instances.json
├── refcocog/
│   ├── refs(umd).p
│   └── instances.json
└── grefcoco/
    ├── grefs(unc).json
    └── instances.json               # 也可以复用 refcoco/instances.json
```

复制示例：

```bash
cp -a playground/data/refer_seg/. PLAS/datasets/refer_seg/
```

### ReasonSeg

```text
PLAS/datasets/ReasonSeg/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

数据集根目录必须包含 `val/` 和 `test/`。推理时通过 `REASONSEG_ROOT=./PLAS/datasets/ReasonSeg` 指定。

## 4. Checkpoint 放置位置

已有权重复制到：

```bash
mkdir -p PLAS/checkpoints/stage1
mkdir -p PLAS/checkpoints/stage2/phaseA_latent_warmup
mkdir -p PLAS/checkpoints/stage2/phaseB_task_adaptation

cp checkpoints/llava6-3-2/stage1/full/mask_latent_teacher.pt \
  PLAS/checkpoints/stage1/mask_latent_teacher.pt
cp checkpoints/llava6-3-8/stage2-1024/phaseA_latent_warmup/qwen_latent_student.pt \
  PLAS/checkpoints/stage2/phaseA_latent_warmup/qwen_latent_student.pt
cp checkpoints/llava6-3-8/stage2-1024/phaseB_task_adaptation/qwen_latent_student.pt \
  PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt
```

如果你的旧目录名称不同，只需要把源文件替换成实际的 Teacher、Phase-A Student 和 Phase-B Student 文件；目标路径保持不变。

## 5. 三阶段训练

以下命令默认使用 4 张 GPU。用 `CUDA_VISIBLE_DEVICES=0` 可以改成单卡调试。

### Stage I：Teacher 预训练

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_STAGE=pipeline \
SAM_PATH=./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
DATA_ROOT=./PLAS/datasets/refer_seg \
OUTPUT_DIR=./PLAS/checkpoints/stage1 \
bash PLAS/scripts/train_stage1_teacher.sh
```

输出：

```text
PLAS/checkpoints/stage1/warmup/mask_latent_teacher.pt
PLAS/checkpoints/stage1/full/mask_latent_teacher.pt
```

### Stage II-A：Latent warmup

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

Stage II-A 使用 Latent Alignment、image-free BCE 和 image-free Dice。

### Stage II-B：Task adaptation

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

Stage II-B 输出：

```text
PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt
```

## 6. 三阶段推理/评估

### Stage I Teacher 重建评估

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
SAM_PATH=./PLAS/pretrained/sam_vit_h/sam_vit_h_4b8939.pth \
DATA_ROOT=./PLAS/datasets/refer_seg \
OUTPUT_DIR=./PLAS/outputs/stage1_teacher \
bash PLAS/scripts/infer_stage1_teacher.sh
```

### Stage II-A RefCOCO 推理

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseA_latent_warmup/qwen_latent_student.pt \
EVAL_SPLIT=val \
OUTPUT_DIR=./PLAS/outputs/refcoco_stage2a \
bash PLAS/scripts/infer_refseg_stage2a.sh
```

### Stage II-B RefCOCO 推理

```bash
cd /data2/lyy/Text4Seg-main
CUDA_VISIBLE_DEVICES=0 \
TEACHER_CHECKPOINT=./PLAS/checkpoints/stage1/full/mask_latent_teacher.pt \
STUDENT_CHECKPOINT=./PLAS/checkpoints/stage2/phaseB_task_adaptation/qwen_latent_student.pt \
EVAL_SPLIT=val \
OUTPUT_DIR=./PLAS/outputs/refcoco_stage2b \
bash PLAS/scripts/infer_refseg_stage2b.sh
```

### Stage II-B ReasonSeg 推理

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

将 `EVAL_SPLIT=test` 可评估 ReasonSeg test。

## 7. 速度测量

速度测试必须单进程单卡，不能使用 `torchrun` 或 `deepspeed`：

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

通过 10 个样本后，将 `MAX_SAMPLES=0` 评估完整验证集。结果保存在 `latency_benchmark_sam.json` 和 `latency_samples_sam.csv`。

## 8. 独立性检查

在仓库根目录执行：

```bash
rg -n "from (llava|qwen)|import llava|llava5_v14_8|llava6-3-8" PLAS --glob '*.py'
```

其中允许出现旧 checkpoint format 字符串和历史注释；不应出现指向其他代码目录的 Python import。当前独立运行路径使用的是 `PLAS.model`、`PLAS.train` 和 `PLAS.eval`。





