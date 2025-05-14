#!/bin/bash
#SBATCH --job-name=llama-lora         # Job name
#SBATCH --partition=gpu_h100          # GPU partition (adjust if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=5gb                    # Request 32 GB of memory
#SBATCH --time=02:00:00               # Runtime limit (4 hours, adjust as needed)
#SBATCH --output=llama_lora_%j.out    # Output file with job ID
#SBATCH --error=llama_lora_%j.err # Error file

# Load the default Python and CUDA modules
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# Check the loaded modules (optional; helpful for debugging)
module list

# Activate your virtual environment
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# Change to your project directory
cd /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune

# Run the fine-tuning script; adjust the path if necessary
python "Data/Python Files/fine_tune_lora.py"
