#!/bin/bash
#SBATCH --job-name=infer_chunked
#SBATCH --output=llama-finetune/chunked_inference_%j.out
#SBATCH --error=llama-finetune/chunked_inference_%j.err
#SBATCH --gres=gpu:1             # request one GPU
#SBATCH --time=00:29:50          # adjust as you like (HH:MM:SS)
#SBATCH --mem=10gb
#SBATCH --cpus-per-task=12

# — load your environment —
module load devel/cuda/12.8
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# — run the chunked inference —
python "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Python Files/inference_chunked.py"
