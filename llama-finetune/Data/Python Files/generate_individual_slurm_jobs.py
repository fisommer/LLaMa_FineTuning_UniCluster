#!/usr/bin/env python3
import os
import stat # For making files executable
import logging

# --- Configuration ---
# Base Paths on UniCluster (for paths *inside* the SLURM scripts)
BASE_PFS_DIR_CLUSTER = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune"
DATA_BASE_DIR_CLUSTER = os.path.join(BASE_PFS_DIR_CLUSTER, "Data")
GENERATED_TEXTS_BASE_DIR_CLUSTER = os.path.join(BASE_PFS_DIR_CLUSTER, "Generated_Texts")
VENV_PATH_CLUSTER = os.path.join(BASE_PFS_DIR_CLUSTER, "venv")

# Base Path for saving the generated .slurm script files (locally or on login node)
LOCAL_SLURM_JOB_SCRIPTS_BASE_DIR = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Jobs"

# Inference script path on cluster
INFERENCE_SCRIPT_PATH_CLUSTER = os.path.join(DATA_BASE_DIR_CLUSTER, "Python_Files", "inference_normal.py")

# SLURM Job Parameters (user-provided values)
DEFAULT_TIME_LIMIT = "00:30:00"
DEFAULT_PARTITION = "dev_gpu_h100"
DEFAULT_GPUS_PER_TASK = "1"
DEFAULT_CPUS_PER_TASK = "8"
DEFAULT_MAIL_TYPE = "FAIL"
DEFAULT_MAIL_USER = "sommerfinn@icloud.com"

# Modules and Environment Activation (from user's fine-tuning job)
MODULE_LOAD_CMDS = """
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
"""
VENV_ACTIVATION_CMD = f"source {os.path.join(VENV_PATH_CLUSTER, 'bin', 'activate')}"

# --- Setup Logging for this generator script ---
# Log for the generator script itself - save it locally or on the login node where you have write access.
# For example, in a "Logs" subdirectory next to where the SLURM scripts are saved.
GENERATOR_SCRIPT_LOG_DIR = os.path.join(LOCAL_SLURM_JOB_SCRIPTS_BASE_DIR, "..", "Logs_Generator") # One level up from "Jobs", then "Logs_Generator"
# This resolves to /Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Logs_Generator
os.makedirs(GENERATOR_SCRIPT_LOG_DIR, exist_ok=True)
LOG_FILE_PATH_GENERATOR = os.path.join(GENERATOR_SCRIPT_LOG_DIR, "slurm_job_generator_individual.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_PATH_GENERATOR, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- SLURM Script Template (using raw string r"""...""") ---
SLURM_SCRIPT_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_log_output_path}
#SBATCH --error={slurm_log_error_path}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --export=ALL
{mail_directives}

# --- SLURM Job Start ---
echo "--- Starting SLURM Job: ${{SLURM_JOB_NAME}} ---"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job ID: ${{SLURM_JOB_ID}}"
echo "GPU(s) assigned: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
echo "Loading modules..."
{module_load_cmds}
echo "Modules loaded."
module list

echo "Activating virtual environment..."
{venv_activation_cmd}
echo "Virtual environment activated. Python: $(which python)"

# --- Define File Paths (these are paths on the cluster for execution) ---
INFERENCE_SCRIPT_PATH="{inference_script_path}"
PROMPT_FILE_PATH="{prompt_file_path}"
OUTPUT_TEXT_FILEPATH="{output_text_filepath}"

# --- Read Prompt Content ---
echo "Reading prompt from: ${{PROMPT_FILE_PATH}}"
if [ ! -f "${{PROMPT_FILE_PATH}}" ]; then
    echo "ERROR: Prompt file not found at ${{PROMPT_FILE_PATH}}"
    exit 1
fi
PROMPT_CONTENT=$(cat "${{PROMPT_FILE_PATH}}")
if [ -z "${{PROMPT_CONTENT}}" ]; then
    echo "ERROR: Prompt file is empty: ${{PROMPT_FILE_PATH}}"
    exit 1
fi
# Escape for shell arguments when passing to python (basic)
# Using single quotes for sed should be safer with complex prompts
ESCAPED_PROMPT_CONTENT_FOR_ARG=$(printf '%s' "$PROMPT_CONTENT" | sed 's/"/\\"/g' | sed 's/\$/\\$/g' | sed 's/`/\\`/g')


echo "Prompt (first 100 chars): $(echo "$PROMPT_CONTENT" | head -c 100)..."
echo "Output will be saved to: ${{OUTPUT_TEXT_FILEPATH}}"

# --- Run Inference ---
echo "Running inference for author style: {author_style_arg}"
python "${{INFERENCE_SCRIPT_PATH}}" \
    --prompt "${{ESCAPED_PROMPT_CONTENT_FOR_ARG}}" \
    --author_style "{author_style_arg}" \
    --output_file "${{OUTPUT_TEXT_FILEPATH}}"

JOB_EXIT_CODE=$?
if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "Inference script completed successfully."
else
    echo "ERROR: Inference script exited with code $JOB_EXIT_CODE."
fi

echo "--- SLURM Job ${{SLURM_JOB_NAME}} Finished ---"
"""

def generate_jobs_for_author(author_config, num_jobs=21):
    author_key_full = author_config["key_full"]
    author_key_simple = author_config["key_simple"]
    author_model_dir_name_cluster = author_config["model_dir_cluster"] # For output on cluster
    author_style_arg = author_config["style_arg"]
    local_slurm_job_subdir_name = author_config["local_slurm_job_subdir"] # New key for local path

    logger.info(f"--- Generating {num_jobs} SLURM job scripts for {author_key_full} ---")

    # Directory where prompt files are located (on cluster)
    prompts_input_dir_cluster = os.path.join(DATA_BASE_DIR_CLUSTER, author_key_full, "Prompts")
    
    # Directory to save the generated .slurm job scripts (locally or on login node)
    slurm_scripts_output_dir_local = os.path.join(LOCAL_SLURM_JOB_SCRIPTS_BASE_DIR, local_slurm_job_subdir_name)
    os.makedirs(slurm_scripts_output_dir_local, exist_ok=True)
    logger.info(f"SLURM job scripts will be saved to (local/login): {slurm_scripts_output_dir_local}")


    # Base directory for generated text outputs (on cluster)
    generated_text_author_dir_cluster = os.path.join(GENERATED_TEXTS_BASE_DIR_CLUSTER, author_model_dir_name_cluster)
    # No need to makedirs here, as the SLURM job itself will run on cluster and inference_normal.py handles it.

    # Base directory for SLURM's own .out and .err logs for these specific jobs (on cluster)
    slurm_stdio_log_author_dir_cluster = os.path.join(GENERATED_TEXTS_BASE_DIR_CLUSTER, "Slurm_Logs_Inference_Individual", author_model_dir_name_cluster)
    # No need to makedirs here for the same reason.

    mail_directives_str = ""
    if DEFAULT_MAIL_USER:
        mail_directives_str = f"#SBATCH --mail-type={DEFAULT_MAIL_TYPE}\n#SBATCH --mail-user={DEFAULT_MAIL_USER}"

    jobs_created_count = 0
    for i in range(1, num_jobs + 1):
        prompt_number_padded = f"{i:02d}"
        job_name = f"GenInf_{author_key_simple[:4].capitalize()}_{prompt_number_padded}"
        
        # Prompt file path (on cluster)
        current_prompt_filename = f"{author_key_simple}_prompt_{prompt_number_padded}.txt"
        prompt_file_path_on_cluster = os.path.join(prompts_input_dir_cluster, current_prompt_filename)

        # Output text file path (on cluster)
        output_text_filename = f"{author_key_simple}_sample_{prompt_number_padded}.txt"
        output_text_filepath_on_cluster = os.path.join(generated_text_author_dir_cluster, output_text_filename)

        # SLURM .out and .err file paths (on cluster)
        slurm_log_output_path_on_cluster = os.path.join(slurm_stdio_log_author_dir_cluster, f"{job_name}.out")
        slurm_log_error_path_on_cluster = os.path.join(slurm_stdio_log_author_dir_cluster, f"{job_name}.err")
        
        # Path where this .slurm script itself will be saved (locally or on login node)
        slurm_job_script_filename = f"run_{job_name}.slurm"
        slurm_job_script_save_path_local = os.path.join(slurm_scripts_output_dir_local, slurm_job_script_filename)

        # Check if the specific prompt file exists *on the cluster path*
        # This check is more of a safeguard; the SLURM script itself will fail if the prompt isn't there.
        # For this generator script, we assume the prompt files *will* exist on the cluster when jobs run.
        # if not os.path.exists(prompt_file_path_on_cluster): # This check would be problematic if run locally
        #     logger.warning(f"Prompt file not found on cluster path: {prompt_file_path_on_cluster}. This check might be inaccurate if run locally. SLURM job will be generated anyway.")
            # continue 

        formatted_script_content = SLURM_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            slurm_log_output_path=slurm_log_output_path_on_cluster,
            slurm_log_error_path=slurm_log_error_path_on_cluster,
            partition=DEFAULT_PARTITION,
            gpus_per_task=DEFAULT_GPUS_PER_TASK,
            time_limit=DEFAULT_TIME_LIMIT,
            cpus_per_task=DEFAULT_CPUS_PER_TASK,
            mail_directives=mail_directives_str,
            module_load_cmds=MODULE_LOAD_CMDS.strip(),
            venv_activation_cmd=VENV_ACTIVATION_CMD,
            inference_script_path=INFERENCE_SCRIPT_PATH_CLUSTER,
            prompt_file_path=prompt_file_path_on_cluster,
            author_style_arg=author_style_arg,
            output_text_filepath=output_text_filepath_on_cluster
        )

        try:
            with open(slurm_job_script_save_path_local, 'w', encoding='utf-8') as f:
                f.write(formatted_script_content)
            os.chmod(slurm_job_script_save_path_local, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            logger.info(f"Successfully created SLURM job script: {slurm_job_script_save_path_local}")
            jobs_created_count += 1
        except Exception as e:
            logger.error(f"Error writing SLURM job script {slurm_job_script_save_path_local}: {e}", exc_info=True)
            
    logger.info(f"Finished generating {jobs_created_count} SLURM job scripts for {author_key_full}.")


def main():
    logger.info("--- Starting SLURM Job Script Generator (Individual Files) ---")

    authors_config = {
        "dickens": {
            "key_full": "Charles_Dickens",
            "key_simple": "dickens",
            "model_dir_cluster": "DickensModel",
            "style_arg": "dickens",
            "local_slurm_job_subdir": "Dickens_Generation"
        },
        "twain": {
            "key_full": "Mark_Twain",
            "key_simple": "marktwain",
            "model_dir_cluster": "TwainModel",
            "style_arg": "twain",
            "local_slurm_job_subdir": "Twain_Generation"
        }
    }

    num_prompts_per_author = 0
    while num_prompts_per_author <= 0:
        try:
            num_str = input(f"How many prompts (SLURM jobs) to generate PER AUTHOR (e.g., 21)? ").strip()
            num_prompts_per_author = int(num_str)
            if num_prompts_per_author <= 0: print("Please enter a positive number.")
        except ValueError: print("Invalid input. Please enter a number.")

    for author_short_name, config in authors_config.items():
        generate_jobs_for_author(config, num_jobs=num_prompts_per_author)
    
    logger.info("--- All SLURM Job Script Generation Finished ---")
    print("\nSLURM job scripts have been generated. You can now navigate to (locally or on login node):")
    for cfg in authors_config.values():
        print(f"  {os.path.join(LOCAL_SLURM_JOB_SCRIPTS_BASE_DIR, cfg['local_slurm_job_subdir'])}")
    print("And submit the jobs using sbatch, respecting your cluster's concurrency limits.")

if __name__ == "__main__":
    main()
