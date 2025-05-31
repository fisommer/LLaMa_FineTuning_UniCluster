#!/bin/bash
#SBATCH --job-name=GenInf_Dick_16
#SBATCH --output=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/Slurm_Logs_Inference_Individual/DickensModel/GenInf_Dick_16.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/Slurm_Logs_Inference_Individual/DickensModel/GenInf_Dick_16.err
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sommerfinn@icloud.com

# --- SLURM Job Start ---
echo "--- Starting SLURM Job: ${SLURM_JOB_NAME} ---"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "GPU(s) assigned: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
echo "Loading modules..."
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
echo "Modules loaded."
module list

echo "Activating virtual environment..."
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate
echo "Virtual environment activated. Python: $(which python)"

# --- Define File Paths (these are paths on the cluster for execution) ---
INFERENCE_SCRIPT_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Python Files/inference_normal.py"
PROMPT_FILE_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Prompts/dickens_prompt_16.txt"
OUTPUT_TEXT_FILEPATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/DickensModel/dickens_sample_16.txt"

# --- Read Prompt Content ---
echo "Reading prompt from: ${PROMPT_FILE_PATH}"
if [ ! -f "${PROMPT_FILE_PATH}" ]; then
    echo "ERROR: Prompt file not found at ${PROMPT_FILE_PATH}"
    exit 1
fi
PROMPT_CONTENT=$(cat "${PROMPT_FILE_PATH}")
if [ -z "${PROMPT_CONTENT}" ]; then
    echo "ERROR: Prompt file is empty: ${PROMPT_FILE_PATH}"
    exit 1
fi
# Escape for shell arguments when passing to python (basic)
# Using single quotes for sed should be safer with complex prompts
ESCAPED_PROMPT_CONTENT_FOR_ARG=$(printf '%s' "$PROMPT_CONTENT" | sed 's/"/\\"/g' | sed 's/\$/\\$/g' | sed 's/`/\\`/g')


echo "Prompt (first 100 chars): $(echo "$PROMPT_CONTENT" | head -c 100)..."
echo "Output will be saved to: ${OUTPUT_TEXT_FILEPATH}"

# --- Run Inference ---
echo "Running inference for author style: dickens"
python "${INFERENCE_SCRIPT_PATH}" \
    --prompt "${ESCAPED_PROMPT_CONTENT_FOR_ARG}" \
    --author_style "dickens" \
    --output_file "${OUTPUT_TEXT_FILEPATH}"

JOB_EXIT_CODE=$?
if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "Inference script completed successfully."
else
    echo "ERROR: Inference script exited with code $JOB_EXIT_CODE."
fi

echo "--- SLURM Job ${SLURM_JOB_NAME} Finished ---"
