#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=test_load_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --mem=128G
#SBATCH --qos=short
#SBATCH --output=result.out
#SBATCH --error=error.log
#SBATCH --mail-type=END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=liming.fan333@gmail.com

# Navigate to your project directory
cd /home/user/eric123/my_project/test

# Load Miniconda
export PATH=~/miniconda3/bin:$PATH

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate py310

# Set PYTHONPATH to include the project directory
export PYTHONPATH=$PYTHONPATH:/home/user/eric123/my_project

# Check GPU availability
echo "Checking GPU availability with nvidia-smi:"
nvidia-smi

# Run your Python script
srun python test_load_data.py

# Deactivate the conda environment
conda deactivate
