#!/bin/bash 

#SBATCH --job-name=full_cohort_run

#SBATCH --partition=normal 

#SBATCH --ntasks=1 

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --output=output/full_cohort_held_out_risk.out 

#SBATCH --error=error/full_cohort_held_out_risk.err

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

python /data/gusev/USERS/jpconnor/clinical_text_project/code/python_scripts/model_evaluation/held_out_ICD10_level_3_risk_scores.py

conda deactivate