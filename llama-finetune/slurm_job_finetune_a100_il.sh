#!/bin/bash
#SBATCH --job-name=llama-lora-a100_il         # Job name
#SBATCH --gres=gpu:1                          # Request 1 GPU on that node
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores
#SBATCH --mem=50gb                            # Request 50GB of memory
#SBATCH --time=06:00:00                       # Set time limit to 6 hours (adjust as needed)
#SBATCH --output=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output_gpu_a100_il/llama_lora_%j.out  # Output file; %j replaced with job ID
#SBATCH --error=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output_gpu_a100_il/llama_lora_%j.err   # Error file

# Load required modules (using the default modules as per the guides)
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# (Optional) List loaded modules for debugging
module list

# Activate the virtual environment
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# Change to the project directory
cd /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune

# Run the fine-tuning script (adjust the path if necessary)
python "Data/Python Files/fine_tune_lora.py"
