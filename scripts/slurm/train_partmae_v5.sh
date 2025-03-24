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
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/scratch-shared/dcanez/HF_HOME"

srun python -m src.main_pretrain experiment=pretrain/in1k/partmae_v5/vit_b_16/h100_single

deactivate
module unload CUDA/12.6.0
module unload 2024