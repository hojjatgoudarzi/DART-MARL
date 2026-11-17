#!/bin/bash

#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_env

# extra flags pass through
cd /nfs/stak/users/negreanj/hpc-share/marl-project
python train.py "$@"
