#!/bin/bash
#SBATCH --job-name=llama-lora-dev         # Job name for development fine-tuning
#SBATCH --gres=gpu:1                     # Request 1 GPU on that node
#SBATCH --cpus-per-task=8                # Request 8 CPU cores
#SBATCH --mem=50gb                       # Request 50GB of memory
#SBATCH --time=00:30:00                  # Set time limit to 30 minutes (HH:MM:SS)
#SBATCH --output=llama_lora_dev_%j.out   # Output file; %j will be replaced with your job ID
#SBATCH --error=llama_lora_dev_%j.err    # Error file

# Load necessary modules using the default versions as per the UniCluster guides:
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# (Optional) Check loaded modules for debugging:
module list

# Activate your virtual environment:
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# Change to your project directory:
cd /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune

# Run the fine-tuning script (this run will only last 30 minutes, which is good for testing).
python "Data/Python Files/fine_tune_lora.py"
