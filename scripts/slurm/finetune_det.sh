#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=train
#SBATCH --output=scripts/slurm_logs/%A.out

module load 2024
module load CUDA/12.6.0

cd $HOME/development/PART
source .venv/bin/activate

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/scratch-shared/dcanez/HF_HOME"
export DETECTRON2_DATASETS="/scratch-shared/dcanez/datasets"

srun python -m src.main_finetune_det --num-gpus=1 --config-file fabric_configs/experiment/detection/mask_rcnn_vitdet_b_12ep_v4_100ep.py 

deactivate
module unload CUDA/12.6.0
module unload 2024