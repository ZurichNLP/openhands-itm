#!/usr/bin/bash -l
#SBATCH --time=4:0:0
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=experiment_scripts/basic_bare/train.out

module load miniforge3
source activate openhands

python experiment_scripts/basic_bare/train.py
