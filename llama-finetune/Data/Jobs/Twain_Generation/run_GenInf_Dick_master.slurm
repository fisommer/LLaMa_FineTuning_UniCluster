#!/bin/bash
#SBATCH --job-name=GenInf_Dickens_All # Master job name for all Dickens prompts
#SBATCH --output=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/Slurm_Logs_Inference_Master/DickensModel/GenInf_Dickens_All.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/Slurm_Logs_Inference_Master/DickensModel/GenInf_Dickens_All.err
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00      # Total time for ALL 21 inferences
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --mail-type=ALL      # Changed to ALL to monitor start/end of the whole batch
#SBATCH --mail-user=sommerfinn@icloud.com

# --- SLURM Job Start ---
echo "--- Starting Master SLURM Job: ${SLURM_JOB_NAME} for Charles Dickens ---"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "GPU(s) assigned: $CUDA_VISIBLE_DEVICES"
echo "Number of prompts to process: 21"

# --- Environment Setup ---
echo "Loading modules..."
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
echo "Modules loaded."
module list

echo "Activating virtual environment..."
VENV_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Virtual environment activated. Python: $(which python)"
else
    echo "ERROR: Virtual environment activation script not found at ${VENV_PATH}/bin/activate"
    exit 1
fi

# --- Define Base Paths (on cluster) ---
INFERENCE_SCRIPT_PATH="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Py Files/inference_normal.py"
PROMPTS_BASE_DIR="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Prompts"
GENERATED_TEXT_BASE_DIR="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts/TwainModel"

# Create output directories if they don't exist
mkdir -p "${GENERATED_TEXT_BASE_DIR}"
mkdir -p "$(dirname "${SLURM_JOB_OUTPUT}")" # SLURM output log directory

# --- Main Inference Loop for Charles Dickens ---
for i in $(seq -f "%02g" 1 21) # Iterates 01, 02, ..., 21
do
    echo "" # Add a blank line for readability in the log
    echo "--- Processing Prompt #${i} for Charles Dickens ---"
    
    # Assuming prompt files are named like dickens_prompt_v2_01.txt, dickens_prompt_v2_02.txt, etc.
    # If your prompt files are named differently (e.g., without "_v2_"), adjust this line:
    CURRENT_PROMPT_FILENAME="dickens_prompt_${i}.txt" 
    CURRENT_PROMPT_FILE_PATH="${PROMPTS_BASE_DIR}/${CURRENT_PROMPT_FILENAME}"
    
    CURRENT_OUTPUT_FILENAME="dickens_sample_${i}.txt"
    CURRENT_OUTPUT_TEXT_FILEPATH="${GENERATED_TEXT_BASE_DIR}/${CURRENT_OUTPUT_FILENAME}"

    echo "Reading prompt from: ${CURRENT_PROMPT_FILE_PATH}"
    if [ ! -f "${CURRENT_PROMPT_FILE_PATH}" ]; then
        echo "ERROR: Prompt file not found: ${CURRENT_PROMPT_FILE_PATH}"
        echo "Skipping this prompt."
        continue # Skip to the next iteration of the loop
    fi
    PROMPT_CONTENT=$(cat "${CURRENT_PROMPT_FILE_PATH}")
    if [ -z "${PROMPT_CONTENT}" ]; then
        echo "ERROR: Prompt file is empty: ${CURRENT_PROMPT_FILE_PATH}"
        echo "Skipping this prompt."
        continue # Skip to the next iteration of the loop
    fi
    
    # Escape for shell arguments when passing to python
    ESCAPED_PROMPT_CONTENT_FOR_ARG=$(printf '%s' "$PROMPT_CONTENT" | sed 's/"/\\"/g' | sed 's/\$/\\$/g' | sed 's/`/\\`/g')

    echo "Prompt (first 100 chars): $(echo "$PROMPT_CONTENT" | head -c 100)..."
    echo "Output text file will be: ${CURRENT_OUTPUT_TEXT_FILEPATH}"

    # Ensure the specific output directory for the inference script exists
    mkdir -p "$(dirname "${CURRENT_OUTPUT_TEXT_FILEPATH}}")"

    python "${INFERENCE_SCRIPT_PATH}" \
        --prompt "${ESCAPED_PROMPT_CONTENT_FOR_ARG}" \
        --author_style "twain" \
        --output_file "${CURRENT_OUTPUT_TEXT_FILEPATH}"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Successfully processed prompt #${i} for Charles Dickens on Mark Twain model."
    else
        echo "ERROR processing prompt #${i} for Charles Dickens. Inference script exited with code $EXIT_CODE."
        # Consider if an error for one prompt should stop the whole job.
        # For now, it continues to the next prompt.
    fi
done

echo ""
echo "--- All Charles Dickens prompts processed for Master SLURM Job ${SLURM_JOB_NAME} ---"