#!/bin/bash
#SBATCH --job-name=llama-lora-h100_il         # Job name
#SBATCH --gres=gpu:1                          # Request 1 GPU on that node
#SBATCH --mem=50gb                            # Request 50GB of memory
#SBATCH --time=04:00:00                       # Set time limit to 4 hours (adjust as needed)
#SBATCH --output=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output_gpu_h100_il/llama_lora_%j.out  # Output file
#SBATCH --error=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output_gpu_h100_il/llama_lora_%j.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sommerfinn@icloud.com

# Load required modules (using the default modules as per the guides)
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# (Optional) Check loaded modules
module list

# Activate the virtual environment
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# Change to the project directory
cd /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune

# Run the fine-tuning script
python "Data/Python Files/fine_tune_lora.py"
