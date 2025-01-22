#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

module load 2024
# module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0

cd $HOME/development/PART
source .venv/bin/activate

╰─❯ python -m src.train hparams_search=im1k_A100 experiment=hs_part_im1k_pairdiff_mlp_A100

deactivate
module unload CUDA/12.6.0
module unload 2024