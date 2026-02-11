#!/bin/bash 

#SBATCH --job-name=full_model_w_stage_run

#SBATCH --partition=normal 

#SBATCH --ntasks=1 

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

#SBATCH --output=output/full_model_run_w_stage.out 

#SBATCH --error=error/full_model_run_w_stage.err

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

python /data/gusev/USERS/jpconnor/clinical_text_project/code/python_scripts/model_training/run_unified_model_w_stage.py

conda deactivate