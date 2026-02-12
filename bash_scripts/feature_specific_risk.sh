#!/bin/bash 

#SBATCH --job-name=full_cohort_run

#SBATCH --partition=normal 

#SBATCH --ntasks=1 

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

#SBATCH --output=output/feature_specific_risk.out 

#SBATCH --error=error/feature_specific_risk.err

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python /data/gusev/USERS/jpconnor/clinical_text_project/code/python_scripts/model_evaluation/feature_ICD10_level_3_risk_scores.py

conda deactivate
