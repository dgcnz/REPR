# REPR: Position-Aware Reconstruction with Transformers

Official implementation of **REPR** (Position-Aware Reconstruction with Transformers) for self-supervised learning.

> **Note**: In this codebase, REPR is referred to as `partmae_v5` (L_pose only) or `partmae_v6` (all loses). 


## Installation

Use [`uv`](https://docs.astral.sh/uv/) package manager to install all dependencies:

```bash
uv sync
```

## Data Setup

### ImageNet-1K
Download ImageNet-1K and organize as:
```
data/
  imagenet/
    train/
      n01440764/
        ...
    val/
      n01440764/
        ...
```

Update paths in `fabric_configs/data/in1k.yaml`:
```yaml
train_root: "/path/to/imagenet/train"
val_root: "/path/to/imagenet/val"
```

### Other Datasets
- **ADE20K**: For semantic segmentation evaluation
- **COCO**: For object detection finetuning  
- **VOC**: For semantic segmentation evaluation

Configure paths in respective config files under `fabric_configs/data/`.

## Pretrained Models

Download pretrained checkpoints from HuggingFace:

```bash
huggingface-cli download dgcnz/REPR --local-dir .
```

Available models:
- `outputs/2025-04-11/10-15-18/epoch_0199.ckpt`: REPR (L_pose only)
- `outputs/2025-06-22/19-16-53/epoch_0199.ckpt`: REPR

## Reproduce Paper Results

### Linear Classification on ImageNet-1K
```bash
uv run python -m src.experiments.linear_classification.main_linear \
    model=partmaev6_ep199_b \
    data=imagenet
```

### Semantic Segmentation on ADE20K
```bash
uv run python -m src.experiments.linear_segmentation.eval_linear \
    model=partmaev6_b_ep199 \
    data=ade20k
```

### Object Detection on COCO
```bash
uv run python -m src.main_finetune_det \
    model=partmaev6_b_ep199 \
    data=coco
```

### K-NN Classification
```bash
uv run python -m src.experiments.knn.main_knn \
    model=partmaev6_b_ep199 \
    data=imagenet
```

## Training from Scratch

### Pretraining

**Local (for debugging):**
```bash
uv run python -m src.main_pretrain \
    experiment=pretrain/in1k/partmae_v6_vit_b_16/4060ti
```

**SLURM cluster:**
```bash
sbatch scripts/slurm/train_partmae_v6_h100.sh
```

### Custom Training
Config files are in `fabric_configs/experiment/`. Override parameters:

```bash
uv run python -m src.main_pretrain \
    experiment=pretrain/in1k/partmae_v6_vit_b_16/4060ti \
    trainer.max_epochs=300 \
    data.batch_size=256
```

## Model Architecture

The main model is implemented in `src/models/components/partmae_v6.py`.

Key features:
- Off-grid position embedding for flexible patch sampling
- Position-aware reconstruction with pose loss
- Multi-crop training support
- Distributed training with PyTorch Lightning Fabric

## Repository Structure

```
├── src/
│   ├── main_pretrain.py          # Main pretraining script
│   ├── models/components/        # Model implementations
│   ├── experiments/             # Evaluation scripts
│   └── data/                    # Data loading and preprocessing
├── fabric_configs/              # Hydra configuration files
├── scripts/slurm/              # SLURM job scripts
└── tests/                      # Unit tests and benchmarks
```

