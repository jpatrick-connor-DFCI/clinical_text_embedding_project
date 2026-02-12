#!/bin/bash
#SBATCH --job-name=vscode_interactive
#SBATCH --partition=normal    # Adjust partition based on availability
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    
#SBATCH --mem=128G         
#SBATCH --time=120:00:00      
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=patrickconnor@g.harvard.edu
#SBATCH --output=output/vscode_job_%j.out
#SBATCH --error=error/vscode_job_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Print the assigned node name
echo "Assigned node: $(hostname)"

# Keep the session alive
sleep infinity
