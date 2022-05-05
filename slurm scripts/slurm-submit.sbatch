#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=ray-job
#SBATCH --output=ray-job-%j.log
#SBATCH --partition=memory
### Select number of nodes
#SBATCH --nodes=1
### Always run exclusive to avoid issues with conflicting ray servers (needs fixing in the future)
#SBATCH --exclusive
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=2GB
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
# Load Delft Blue modules here
# NOTE: git and openssh are included to not break dependencies 
# NOTE: python is not required if it is included in the conda environment. 
# If python is needed, it must be loaded before miniconda to avoid issues with conda activation (module conda not found within internal python calls)

module load git openssh 
declare -x RAY_TIMELINE="1"

# Adapt this to your job requirements
. ray-on-conda-on-slurm.sh \
    --python_runfile="DistGanRay.py" \
    --python_arguments="" \
    --conda_env="raytest" \
    --temp_dir="/scratch/${USER}/ray"
