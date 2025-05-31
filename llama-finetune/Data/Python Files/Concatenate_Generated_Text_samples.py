#!/usr/bin/env python3
import os
import glob
import re
import logging

# --- Configuration ---
BASE_GENERATED_TEXTS_DIR = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Generated_Texts"

DICKENS_SAMPLES_DIR = os.path.join(BASE_GENERATED_TEXTS_DIR, "DickensModel")
TWAIN_SAMPLES_DIR = os.path.join(BASE_GENERATED_TEXTS_DIR, "TwainModel")

# Output configuration for individual author concatenations
DICKENS_CONCAT_SUBDIR = "Concatenated_Dickens_Only" # New subdirectory for Dickens output
DICKENS_CONCAT_FILENAME = "all_dickens_model_samples.txt"

TWAIN_CONCAT_SUBDIR = "Concatenated_Twain_Only" # New subdirectory for Twain output
TWAIN_CONCAT_FILENAME = "all_twain_model_samples.txt"

# Log file configuration (centralized for this script's operations)
CONCATENATION_LOG_BASE_DIR = BASE_GENERATED_TEXTS_DIR 
CONCATENATION_LOG_SUBDIR = "Script_Logs" # New subdirectory for this script's logs
CONCATENATION_LOG_FILENAME = "concatenation_process_author_specific.log"

NUM_SAMPLES_EXPECTED_PER_AUTHOR = 21 

# --- Setup Logging ---
LOG_OUTPUT_DIR = os.path.join(CONCATENATION_LOG_BASE_DIR, CONCATENATION_LOG_SUBDIR)
os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_OUTPUT_DIR, CONCATENATION_LOG_FILENAME)

# Clear existing handlers from the root logger to avoid duplicate logs if script is re-run
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def natural_sort_key(filename):
    """
    Key for sorting filenames like 'name_01.txt', 'name_02.txt', 'name_10.txt' correctly.
    """
    return [int(num) if num.isdigit() else num.lower() for num in re.split(r'(\d+)', filename)]

def concatenate_author_samples(author_name, samples_dir, file_pattern, 
                               output_subdir_name, output_filename,
                               num_expected_samples):
    """
    Concatenates all sample files for a specific author into a single output file.
    """
    logger.info(f"--- Starting concatenation for {author_name} ---")
    
    full_pattern = os.path.join(samples_dir, file_pattern)
    sample_files = sorted(glob.glob(full_pattern), key=natural_sort_key)
    
    logger.info(f"Found {len(sample_files)} {author_name} sample files in {samples_dir} matching '{file_pattern}'")
    if not sample_files:
        logger.warning(f"No {author_name} files found. Skipping concatenation for this author.")
        return
    if len(sample_files) < num_expected_samples:
        logger.warning(f"Expected {num_expected_samples} {author_name} files, found {len(sample_files)}.")

    # Output path for this author's concatenated file
    # Saved within the author's specific model directory, in a new subdirectory
    author_concat_output_dir = os.path.join(samples_dir, output_subdir_name)
    os.makedirs(author_concat_output_dir, exist_ok=True)
    concatenated_filepath = os.path.join(author_concat_output_dir, output_filename)
    
    logger.info(f"{author_name} concatenated output will be saved to: {concatenated_filepath}")

    try:
        with open(concatenated_filepath, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(sample_files):
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                    
                    outfile.write(content.strip()) 
                    logger.info(f"Appended: {file_path}")
                except Exception as e:
                    logger.error(f"Error reading or writing file {file_path}: {e}")
        
        logger.info(f"Successfully concatenated {len(sample_files)} {author_name} samples to {concatenated_filepath}")

    except Exception as e:
        logger.error(f"An error occurred during {author_name} concatenation process: {e}", exc_info=True)


def main():
    logger.info("--- Starting Concatenation of Generated Text Samples (Author-Specific) ---")

    # Concatenate Dickens Samples
    concatenate_author_samples(
        author_name="Dickens",
        samples_dir=DICKENS_SAMPLES_DIR,
        file_pattern="dickens_sample_*.txt", 
        output_subdir_name=DICKENS_CONCAT_SUBDIR,
        output_filename=DICKENS_CONCAT_FILENAME,
        num_expected_samples=NUM_SAMPLES_EXPECTED_PER_AUTHOR
    )

    logger.info("-" * 50) 

    # Concatenate Twain Samples
    concatenate_author_samples(
        author_name="Twain",
        samples_dir=TWAIN_SAMPLES_DIR,
        file_pattern="marktwain_sample_*.txt", # Ensure this matches your Twain sample filenames
        output_subdir_name=TWAIN_CONCAT_SUBDIR,
        output_filename=TWAIN_CONCAT_FILENAME,
        num_expected_samples=NUM_SAMPLES_EXPECTED_PER_AUTHOR
    )

    logger.info("--- All Concatenation Processes Finished ---")

if __name__ == "__main__":
    main()