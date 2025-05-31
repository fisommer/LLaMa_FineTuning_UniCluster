#!/bin/bash

# Define base paths
SCRIPT_DIR="/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Python_Files" # Assuming inference_normal.py is here
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference_normal.py" # Your modified inference script

DICKENS_PROMPTS_FILE="${SCRIPT_DIR}/dickens_prompts.txt" # Output from prompt extraction
TWAIN_PROMPTS_FILE="${SCRIPT_DIR}/twain_prompts.txt"   # Output from prompt extraction

# Output directory for generated samples
DICKENS_OUTPUT_DIR="/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Generated_Texts/DickensModel"
TWAIN_OUTPUT_DIR="/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Generated_Texts/TwainModel"

mkdir -p "$DICKENS_OUTPUT_DIR"
mkdir -p "$TWAIN_OUTPUT_DIR"

# SLURM job parameters (adjust as needed for UniCluster)
JOB_NAME_PREFIX="GenText"
TIME_LIMIT="00:15:00" # 15 minutes per inference job should be safe
PARTITION="gpu_h100" # Or your specific GPU partition
GPUS_PER_TASK="1"
MEM_PER_GPU="80G" # Or what you need

# Generate for Dickens Prompts (using Dickens-tuned model)
counter=1
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "$prompt" ]]; then continue; fi # Skip empty lines
  sbatch --job-name="${JOB_NAME_PREFIX}_D_${counter}" \
         --time="$TIME_LIMIT" \
         --partition="$PARTITION" \
         --gres="gpu:$GPUS_PER_TASK" \
         --mem-per-gpu="$MEM_PER_GPU" \
         --output="${DICKENS_OUTPUT_DIR}/slurm_dickens_gen_${counter}.out" \
         --error="${DICKENS_OUTPUT_DIR}/slurm_dickens_gen_${counter}.err" \
         <<EOF
#!/bin/bash
echo "Running Dickens generation job ${counter}"
echo "Prompt: ${prompt}"
python "${INFERENCE_SCRIPT}" \
    --prompt "${prompt}" \
    --author_style "dickens" \
    --output_file "${DICKENS_OUTPUT_DIR}/dickens_sample_${counter}.txt"
echo "Finished Dickens generation job ${counter}"
EOF
  counter=$((counter+1))
done < "$DICKENS_PROMPTS_FILE"

# Generate for Twain Prompts (using Twain-tuned model)
counter=1
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "$prompt" ]]; then continue; fi # Skip empty lines
  sbatch --job-name="${JOB_NAME_PREFIX}_T_${counter}" \
         --time="$TIME_LIMIT" \
         --partition="$PARTITION" \
         --gres="gpu:$GPUS_PER_TASK" \
         --mem-per-gpu="$MEM_PER_GPU" \
         --output="${TWAIN_OUTPUT_DIR}/slurm_twain_gen_${counter}.out" \
         --error="${TWAIN_OUTPUT_DIR}/slurm_twain_gen_${counter}.err" \
         <<EOF
#!/bin/bash
echo "Running Twain generation job ${counter}"
echo "Prompt: ${prompt}"
python "${INFERENCE_SCRIPT}" \
    --prompt "${prompt}" \
    --author_style "twain" \
    --output_file "${TWAIN_OUTPUT_DIR}/twain_sample_${counter}.txt"
echo "Finished Twain generation job ${counter}"
EOF
  counter=$((counter+1))
done < "$TWAIN_PROMPTS_FILE"

echo "All generation jobs submitted."