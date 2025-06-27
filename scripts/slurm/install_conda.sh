#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=train
#SBATCH --output=scripts/slurm_logs/%A.out

module load 2024
module load CUDA/12.6.0
module load Anaconda3/2024.06-1

cd $HOME/development/PART

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/scratch-shared/dcanez/HF_HOME"
export TORCHINDUCTOR_CACHE_DIR="/scratch-local/dcanez/tmp/torchinductor/"
export HYDRA_FULL_ERROR=1 

srun conda env create -f env.yaml

module unload Anaconda3/2024.06-1
module unload CUDA/12.6.0
module unload 2024