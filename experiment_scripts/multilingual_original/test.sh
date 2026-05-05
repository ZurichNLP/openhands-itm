#!/usr/bin/bash -l
#SBATCH --time=10:0:0
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=experiment_scripts/multilingual_original/test_labels.out

module load miniforge3
source activate openhands
python experiment_scripts/multilingual_original/test.py
