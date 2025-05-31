#!/bin/bash
#SBATCH --job-name=GenInf_Dick_01
#SBATCH --output=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/Slurm_Logs_Inference_Individual/DickensModel/GenInf_Dick_01.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/Slurm_Logs_Inference_Individual/DickensModel/GenInf_Dick_01.err
#SBATCH --partition=dev_gpu_h100 # VERIFY: Ensure this is the correct partition name (e.g., dev_gpu_h or a production H100 partition)
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00      # 15 minutes should be sufficient for one inference
#SBATCH --cpus-per-task=8    # Number of CPUs
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL     # Optional: Or ALL, NONE
#SBATCH --mail-user=sommerfinn@icloud.com # Optional: Your email

# --- SLURM Job Start ---
echo "--- Starting SLURM Job: ${SLURM_JOB_NAME} ---"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID (if applicable): ${SLURM_ARRAY_TASK_ID}"
echo "GPU(s) assigned: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
echo "Loading modules..."
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
echo "Modules loaded."
module list # List loaded modules for debugging

echo "Activating virtual environment..."
VENV_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Virtual environment activated. Python: $(which python)"
else
    echo "ERROR: Virtual environment activation script not found at ${VENV_PATH}/bin/activate"
    exit 1
fi

# --- Define File Paths ---
INFERENCE_SCRIPT_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Python_Files/inference_normal.py"
PROMPT_FILE_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Prompts/dickens_prompt_01.txt" # Specific to this job
OUTPUT_TEXT_FILEPATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/DickensModel/dickens_sample_01.txt" # Specific to this job

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
# Basic escaping for shell arguments (Python script should handle further if needed)
ESCAPED_PROMPT_CONTENT=$(printf '%s\n' "$PROMPT_CONTENT" | sed "s/[\"\$\`]/\\\\&/g")

echo "Prompt (first 100 chars): ${PROMPT_CONTENT:0:100}..."
echo "Output will be saved to: ${OUTPUT_TEXT_FILEPATH}"

# --- Run Inference ---
echo "Running inference for author style: dickens"
python "${INFERENCE_SCRIPT_PATH}" \
    --prompt "${ESCAPED_PROMPT_CONTENT}" \
    --author_style "dickens" \
    --output_file "${OUTPUT_TEXT_FILEPATH}"

JOB_EXIT_CODE=$?
if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "Inference script completed successfully."
else
    echo "ERROR: Inference script exited with code $JOB_EXIT_CODE."
fi

echo "--- SLURM Job ${SLURM_JOB_NAME} Finished ---"
