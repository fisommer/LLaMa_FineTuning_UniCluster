#!/bin/bash

# --- UniCluster Path Configuration ---
CLUSTER_BASE_DIR="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data"
PYTHON_SCRIPT_DIR="${CLUSTER_BASE_DIR}/Python_Files"
INFERENCE_SCRIPT="${PYTHON_SCRIPT_DIR}/inference_normal.py" # Your modified inference Python script

DICKENS_PROMPTS_FILE="${CLUSTER_BASE_DIR}/Charles_Dickens/Prompts/charlesdickens_prompts_v2_21.txt"
TWAIN_PROMPTS_FILE="${CLUSTER_BASE_DIR}/Mark_Twain/Prompts/marktwain_prompts_v2_21.txt"

GENERATED_TEXTS_BASE_DIR="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Generated_Texts"
DICKENS_MODEL_OUTPUT_DIR="${GENERATED_TEXTS_BASE_DIR}/DickensModel"
TWAIN_MODEL_OUTPUT_DIR="${GENERATED_TEXTS_BASE_DIR}/TwainModel"

SLURM_LOG_BASE_DIR="${GENERATED_TEXTS_BASE_DIR}/Slurm_Logs_Inference" # Separate log dir for inference
DICKENS_SLURM_LOG_DIR="${SLURM_LOG_BASE_DIR}/DickensModel"
TWAIN_SLURM_LOG_DIR="${SLURM_LOG_BASE_DIR}/TwainModel"

mkdir -p "$DICKENS_MODEL_OUTPUT_DIR"
mkdir -p "$TWAIN_MODEL_OUTPUT_DIR"
mkdir -p "$DICKENS_SLURM_LOG_DIR"
mkdir -p "$TWAIN_SLURM_LOG_DIR"

# --- SLURM Job Parameters ---
JOB_NAME_PREFIX="GenTextInf" # Changed prefix to distinguish from fine-tuning
TIME_LIMIT="00:30:00"        # 15 minutes per inference job
PARTITION="dev_gpu_h100"         # **VERIFY THIS PARTITION NAME FOR UNICLUSTER**
GPUS_PER_TASK="1"
# MEM_PER_GPU="80G" # Often not needed if --gres=gpu:1 includes memory, or use --mem
CPUS_PER_TASK="8" # Match your fine-tuning job for consistency if helpful
MAIL_TYPE="FAIL" # Optional: Get notified only on FAIL to reduce emails
MAIL_USER="sommerfinn@icloud.com" # Optional: Your email

# --- Environment Setup Commands for --wrap ---
# These will be included in each sbatch --wrap command
ENV_SETUP_CMDS="echo 'Loading modules...'; \
module load devel/python/3.12.3-gnu-14.2; \
module load devel/cuda/12.8; \
echo 'Activating virtual environment...'; \
source /pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/venv/bin/activate; \
echo 'Environment set up.'"

echo "📝 Submitting Dickens text generation jobs..."
counter=1
if [ ! -f "$DICKENS_PROMPTS_FILE" ]; then
    echo "Error: Dickens prompts file not found at $DICKENS_PROMPTS_FILE"
    exit 1
fi

while IFS= read -r prompt || [ -n "$prompt" ]; do
  if [[ -z "$prompt" ]]; then continue; fi

  sbatch --job-name="${JOB_NAME_PREFIX}_D_${counter}" \
         --time="$TIME_LIMIT" \
         --partition="$PARTITION" \
         --gres="gpu:${GPUS_PER_TASK}" \
         --cpus-per-task="$CPUS_PER_TASK" \
         --output="${DICKENS_SLURM_LOG_DIR}/slurm_dickens_gen_${counter}.out" \
         --error="${DICKENS_SLURM_LOG_DIR}/slurm_dickens_gen_${counter}.err" \
         $( [[ -n "$MAIL_USER" ]] && echo "--mail-type=$MAIL_TYPE --mail-user=$MAIL_USER" ) \
         --wrap="${ENV_SETUP_CMDS} \
                 echo 'Running Dickens generation job ${counter} for prompt starting: ${prompt:0:50}...'; \
                 python \"${INFERENCE_SCRIPT}\" \
                     --prompt \"${prompt}\" \
                     --author_style \"dickens\" \
                     --output_file \"${DICKENS_MODEL_OUTPUT_DIR}/dickens_sample_${counter}.txt\"; \
                 echo 'Finished Dickens generation job ${counter}'"
  
  echo "Submitted job for Dickens prompt #${counter}"
  counter=$((counter+1))
done < "$DICKENS_PROMPTS_FILE"

echo -e "\n📝 Submitting Twain text generation jobs..."
counter=1
if [ ! -f "$TWAIN_PROMPTS_FILE" ]; then
    echo "Error: Twain prompts file not found at $TWAIN_PROMPTS_FILE"
    exit 1
fi

while IFS= read -r prompt || [ -n "$prompt" ]; do
  if [[ -z "$prompt" ]]; then continue; fi

  sbatch --job-name="${JOB_NAME_PREFIX}_T_${counter}" \
         --time="$TIME_LIMIT" \
         --partition="$PARTITION" \
         --gres="gpu:${GPUS_PER_TASK}" \
         --cpus-per-task="$CPUS_PER_TASK" \
         --output="${TWAIN_SLURM_LOG_DIR}/slurm_twain_gen_${counter}.out" \
         --error="${TWAIN_SLURM_LOG_DIR}/slurm_twain_gen_${counter}.err" \
         $( [[ -n "$MAIL_USER" ]] && echo "--mail-type=$MAIL_TYPE --mail-user=$MAIL_USER" ) \
         --wrap="${ENV_SETUP_CMDS} \
                 echo 'Running Twain generation job ${counter} for prompt starting: ${prompt:0:50}...'; \
                 python \"${INFERENCE_SCRIPT}\" \
                     --prompt \"${prompt}\" \
                     --author_style \"twain\" \
                     --output_file \"${TWAIN_MODEL_OUTPUT_DIR}/twain_sample_${counter}.txt\"; \
                 echo 'Finished Twain generation job ${counter}'"

  echo "Submitted job for Twain prompt #${counter}"
  counter=$((counter+1))
done < "$TWAIN_PROMPTS_FILE"

echo -e "\n🎉 All generation jobs submitted."