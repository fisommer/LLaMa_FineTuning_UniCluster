#!/usr/bin/env python3
import os
import glob
import re
import shutil
import logging

# ── ASK USER WHICH AUTHOR TO CLEAN ─────────────────────────────────────────
choice = input("Which data should I clean? [d]ickens or [t]wain: ").strip().lower()
if choice.startswith("d"):
    author_key  = "Charles_Dickens"
    src_folder  = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}"
    dest_folder = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Cleaned"
    log_folder = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Log Files"
elif choice.startswith("t"):
    author_key  = "Mark_Twain"
    src_folder  = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}"
    dest_folder = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Cleaned"
    log_folder = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Log Files"
else:
    print("Unrecognized choice, exiting.")
    exit(1)

# ── SETUP LOGGING ───────────────────────────────────────────────────────────
# Create log directory if it doesn't exist.
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "cleaning_data.log")

logger = logging.getLogger("cleaning_data")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(fmt)
fh.setFormatter(fmt)

logger.addHandler(ch)
logger.addHandler(fh)

logger.info(f"Starting cleaning for {author_key}")
logger.info(f"Source folder:      {src_folder}")
logger.info(f"Destination folder: {dest_folder}")
logger.info(f"Log file:           {log_file}")

# Create the destination folder if it doesn't exist.
os.makedirs(dest_folder, exist_ok=True)

# Find all .txt files recursively in the source folder.
files = glob.glob(os.path.join(src_folder, "**", "*.txt"), recursive=True)
logger.info(f"Found {len(files)} text files to process.")

# Define markers and regex for extracting the title.
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_marker   = "*** END OF THE PROJECT GUTENBERG"
start_pattern = re.compile(
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK\s*(.*?)\s*\*\*\*",
    re.IGNORECASE | re.DOTALL
)

# Process each file.
for file_path in files:
    logger.debug(f"Processing {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Could not read file {file_path}: {e}")
        continue

    start_index = text.find(start_marker)
    if start_index == -1:
        logger.warning(f"Start marker not found in {file_path}, skipping.")
        continue

    match = start_pattern.search(text)
    if match:
        title = match.group(1).strip()
    else:
        logger.warning(f"Could not extract title via regex in {file_path}, skipping.")
        continue

    # Sanitize title for filename
    safe_title = re.sub(r"[^\w\s-]", "", title).strip()
    safe_title = re.sub(r"\s+", "_", safe_title)
    if not safe_title:
        logger.warning(f"Sanitized title empty for {file_path}, skipping.")
        continue

    # Remove header
    header_end = text.find("\n", start_index)
    header_end = header_end + 1 if header_end != -1 else (start_index + len(start_marker))
    cleaned_text = text[header_end:]

    # Truncate at end marker
    end_index = cleaned_text.find(end_marker)
    if end_index != -1:
        cleaned_text = cleaned_text[:end_index]
    else:
        logger.warning(f"End marker not found in {file_path}, keeping full text after header.")

    dest_file = os.path.join(dest_folder, f"{safe_title}.txt")
    try:
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info(f"Cleaned and saved: {dest_file}")
    except Exception as e:
        logger.error(f"Could not write cleaned file {dest_file}: {e}")
        continue

    # Optionally delete original
    try:
        os.remove(file_path)
        logger.debug(f"Deleted original: {file_path}")
    except Exception as e:
        logger.warning(f"Could not delete original {file_path}: {e}")

logger.info("Cleaning process complete.")
