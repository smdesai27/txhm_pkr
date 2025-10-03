#!/bin/bash

# Define job requirements
#SBATCH --job-name=torch_job        # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2           # Number of CPU cores per task
#SBATCH --mem=16G                   # Total memory per node
#SBATCH --time=45:00:00             # Wall time (hh:mm:ss)

# Output and Error logs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load the necessary modules for the GPU environment
module load cuda                      # Load the CUDA toolkit module

# Your job commands
source /users/smdesai/txhm_pkr/pytorch.venv/bin/activate
python /users/smdesai/txhm_pkr/Train.py