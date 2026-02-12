#!/bin/bash 

#SBATCH --job-name=mortality_trajectories

#SBATCH --partition=normal 

#SBATCH --ntasks=1 

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --mail-user=patrickconnor@g.harvard.edu

#SBATCH --output=output/mortality_trajectories.out 

#SBATCH --error=error/mortality_trajectories.err

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python /data/gusev/USERS/jpconnor/clinical_text_project/code/python_scripts/mortality_trajectories/generate_mortality_trajectories.py

conda deactivate
