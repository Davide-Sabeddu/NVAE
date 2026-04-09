#!/bin/bash
#SBATCH --job-name=NVAE
#SBATCH --output=logs/NVAE_%j.out
#SBATCH --error=logs/NVAE_%j.err
#SBATCH --time=54:00:00
#SBATCH --partition=highgpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davide.sabeddu@ucf.edu
#SBATCH --export=NONE

mkdir -p logs

# Move to NVAE root so all local imports (model, utils, datasets…) resolve
cd /home/da389032/NVAE

# Check CUDA availability
~/.conda/envs/sips/bin/python -c "import torch; print(torch.cuda.is_available())"

# Run training — all hyperparameters come from configs/nvae_cifar10.json
~/.conda/envs/sips/bin/python scripts/train_plus.py --phase train