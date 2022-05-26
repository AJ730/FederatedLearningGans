#!/bin/sh
#SBATCH --job-name="sol"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks-per-node=2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


module load 2022r1
module load gpu
module load cuda/11.1.1-zo3qpgx
module load openmpi/4.1.1-i2mystw
module load py-tensorflow/2.4.1-b6kuk57
module load py-pip/21.1.2-zxgv7pz
module load python/3.8.12-bohr45d
module load py-matplotlib/3.4.3-htvky5w
module load py-keras-preprocessing/1.1.2-276l6cy
srun python OfflineGanTensorflow.py
