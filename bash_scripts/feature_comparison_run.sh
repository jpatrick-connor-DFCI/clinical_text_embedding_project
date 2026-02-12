#!/bin/bash 

#SBATCH --job-name=full_model_w_stage_run

#SBATCH --partition=normal 

#SBATCH --ntasks=1 

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

#SBATCH --output=output/feature_comps/batch_mets.out 

#SBATCH --error=error/feature_comps/batch_mets.err

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python /data/gusev/USERS/jpconnor/clinical_text_project/code/python_scripts/model_training/feature_comps_batches/batch_mets.py

conda deactivate
