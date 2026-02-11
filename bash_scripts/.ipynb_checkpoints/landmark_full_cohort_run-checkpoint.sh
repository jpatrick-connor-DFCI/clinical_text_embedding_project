#!/bin/bash 

#SBATCH --job-name=full_cohort_run

#SBATCH --partition=normal 

#SBATCH --ntasks=1 

#SBATCH --cpus-per-task=16

#SBATCH --mem=64G

#SBATCH --output=output/landmark_full_cohort_run/12_months/batch_mets.out 

#SBATCH --error=error/landmark_full_cohort_run/12_months/batch_mets.err

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

python /data/gusev/USERS/jpconnor/clinical_text_project/code/python_scripts/model_training/landmark_full_cohort/12_month/batch_mets.py

conda deactivate