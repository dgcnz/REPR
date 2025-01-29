#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=4   
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --job-name=train
#SBATCH --output=scripts/slurm_logs/%A.out

module load 2024
module load CUDA/12.6.0

cd $HOME/development/PART
source .venv/bin/activate

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_LOGS=dynamo
export PYTHONFAULTHANDLER=1

srun python -m src.train experiment=classification_ft/vit_b_16_imagenet_A100

deactivate
module unload CUDA/12.6.0
module unload 2024
