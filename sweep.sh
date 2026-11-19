#!/bin/bash

#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --job-name=sweep
#SBATCH --output=checkpoints/slurm-sweep-%A_%a.out
#SBATCH --mail-type=END,FAIL

# Communication-radius sweep as one job array: task i trains radius RADII[i] with the default seed.
# All runs go to checkpoints/sweep_<array job id>/, one folder per run.
#
#   sbatch sweep.sh                                   # radii 0 8 15 inf (4 tasks)
#   sbatch sweep.sh --updates 1000                    # extra flags go to every train.py run
#   RADII="8 inf" sbatch --array=0-1 sweep.sh         # other radii: --array=0-(#radii - 1)

RADII=(${RADII:-0 8 15 inf})
r=${RADII[$SLURM_ARRAY_TASK_ID]}

# conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_env

# extra flags pass through
cd /nfs/stak/users/negreanj/hpc-share/marl-project
python train.py --comm-radius "$r" --checkpoint-dir "checkpoints/sweep_${SLURM_ARRAY_JOB_ID}" "$@"
