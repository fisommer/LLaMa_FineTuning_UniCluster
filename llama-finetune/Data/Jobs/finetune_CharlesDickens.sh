#!/bin/bash
#SBATCH --job-name=ft_dickens_full
#SBATCH --output="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Log Files/slurm_dickens_full.out"
#SBATCH --error="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Log Files/slurm_dickens_full.err"
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=00:30:00          # wall-time limit of 30 min
#SBATCH --nodes=1                # single node
#SBATCH --ntasks=1               # one task
#SBATCH --ntasks-per-node=1      # max tasks per node
#SBATCH --cpus-per-task=8        # CPUs per task for data loading
#SBATCH --export=ALL             # propagate environment variables
#SBATCH --mail-type=ALL          # Send email on all events (BEGIN, END, FAIL)
#SBATCH --mail-user=sommerfinn@icloud.com # Email address for notifications

# Load Python and CUDA
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# Show modules for debugging
module list

# Activate your virtual environment
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# Change to the directory containing your full fine-tune script
cd /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Python\ Files

# Run the full Dickens fine‐tuning
python fine_tune_lora_Dickens.py
