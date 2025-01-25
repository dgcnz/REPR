#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=4   
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --job-name=train
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

module load 2024
module load CUDA/12.6.0

cd $HOME/development/PART
source .venv/bin/activate

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python -m src.train experiment=hs_part_im1k_pairdiff_mlp_A100 trainer.num_nodes=1 data.batch_size=256

deactivate
module unload CUDA/12.6.0
module unload 2024
