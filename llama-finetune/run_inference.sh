#!/bin/bash
#SBATCH --job-name=llama_infer
#SBATCH --output=chunked_%j.out
#SBATCH --error=chunked_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=5gb
#SBATCH --gres=gpu:1
#SBATCH --time=0:29:00

# activate environment
module load devel/cuda/12.8
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate

# paths
BASE_MODEL="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
ADAPTER_MODEL="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output"
SCRIPT="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Python Files/inference_chunked.py"
OUTPUT_FILE="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/generated_55000.txt"

# run inference
srun python3 "$SCRIPT" \
  --base-model-path "$BASE_MODEL" \
  --adapter-path "$ADAPTER_MODEL" \
  --prompt "It was the best of times, it was the worst of times..." \
  --total-tokens 55189 \
  --chunk-size 55189 \
  --out-file "$OUTPUT_FILE"
