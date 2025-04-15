#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
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
export TORCHINDUCTOR_CACHE_DIR="/scratch-local/dcanez/tmp/torchinductor/"


srun python -m src.main_pretrain \
    --config-path /gpfs/home2/dcanez/development/PART/outputs/2025-04-11/10-15-16/.hydra \
    --config-name config.yaml \
    ckpt_path=/gpfs/home2/dcanez/development/PART/outputs/2025-04-11/10-15-16/epoch_0199.ckpt \
    +model.freeze_encoder=true \
    "~logger.wandb.id" \
    "~scheduler" \
    data.transform.n_global_crops=2 \
    data.transform.n_local_crops=0 \
    trainer.max_epochs=300 \
    optimizer.lr=1e-6

deactivate
module unload CUDA/12.6.0
module unload 2024

