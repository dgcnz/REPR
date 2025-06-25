#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
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
export TORCHINDUCTOR_CACHE_DIR="/scratch-local/dcanez/tmp/torchinductor/"
export HYDRA_FULL_ERROR=1

srun  python -m src.experiments.linear_segmentation.linear_finetune \
	data=voc \
	'data.data_dir=/gpfs/scratch1/shared/dcanez/datasets/voc' \
	seed=0 \
	model=partmaev6_b_ep199 \
	+precision=bf16-mixed  \
	+fp32=high


deactivate
module unload CUDA/12.6.0
module unload 2024