#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=ray-job
#SBATCH --output=ray-job-%j.log
#SBATCH --partition=compute
### Select number of nodes
#SBATCH --nodes=1
### Always run exclusive to avoid issues with conflicting ray servers
### See also https://github.com/ray-project/ray/issues/10154
#SBATCH --exclusive
#SBATCH --cpus-per-task=44
#SBATCH --mem-per-cpu=2GB
### Give all resources to a single Ray task; ray will manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=0
#SBATCH --time=24:00:00

# Author: Simon Tindemans, s.h.tindemans@tudelft.nl
# Version: 30 April 2022

# Load Delft Blue modules here
# NOTE: git and openssh are included to not break dependencies for on-the-fly installs
# NOTE: python is not required if it is included in the conda environment. 
# The order of loading python and miniconda may be important.


# Option: save timeline info for debugging (yes="1", no="0"); see https://docs.ray.io/en/latest/ray-core/troubleshooting.html
declare -x RAY_TIMELINE="0"

# Adapt this to your job requirements
python_runfile="DistributedGanTensorflow.py"
python_arguments=""
conda_env="rayJobs"
run_dir="../"
temp_dir="/scratch/${USER}/ray"


# DO NOT EDIT HERE
. ray-on-conda-on-slurm.sh \
    --python_runfile="${python_runfile}" \
    --python_arguments="${python_arguments}" \
    --conda_env="${conda_env}" \
    --run_dir="${run_dir}" \
    --temp_dir="${temp_dir}"