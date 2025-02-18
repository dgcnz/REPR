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

cd third-party/DropPos/
srun python -m torch.distributed.run --standalone --nproc-per-node=4 --nnodes 1  main_finetune.py --batch_size 128 --accum_iter 8 --model vit_base_patch16 --finetune ../../artifacts/model-2knf0d16:v0/backbone.ckpt --epochs 100 --warmup_epochs 5 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path hf+ILSVRC/imagenet-1k  --nb_classes 1000 --output_dir ./output_dir --log_dir ./log_dir --experiment finetune_partmae_in1k

deactivate
module unload CUDA/12.6.0
module unload 2024


