#!/bin/bash
#SBATCH --job-name=llama_infer_normal
#SBATCH --output=chunked_%j.out
#SBATCH --error=chunked_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=5gb
#SBATCH --gres=gpu:1
#SBATCH --time=0:29:00

# activate environment
module load devel/cuda/12.8
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# run inference
python /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Python\ Files/inference_normal.py
