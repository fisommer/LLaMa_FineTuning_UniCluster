#!/usr/bin/env python3
"""
Collect all Gutenberg texts for a chosen author, log successes and failures,
and write summary files (titles collected, missing titles) plus a log.
"""
import os
import pandas as pd
import shutil
import logging

# ── ASK WHICH AUTHOR TO COLLECT ─────────────────────────────────────────────
choice = input("Which author’s books shall I collect? [d]ickens or [t]wain: ").strip().lower()
if choice.startswith("d"):
    AUTHOR_KEY = "Dickens, Charles"
    destination_folder = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Charles_Dickens"
elif choice.startswith("t"):
    AUTHOR_KEY = "Twain, Mark"
    destination_folder = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Mark_Twain"
else:
    print("✖️  Unrecognized choice, exiting.")
    exit(1)

print(f"→ collecting all ‘{AUTHOR_KEY}’ (English) into {destination_folder}")

# Ensure destination exists
os.makedirs(destination_folder, exist_ok=True)

# Set up log directory and files
log_folder = os.path.join(destination_folder, "Log Files")
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "collect_data.log")
collected_file = os.path.join(log_folder, "collected_titles.txt")
missing_file = os.path.join(log_folder, "missing_titles.txt")

# Configure logging
def setup_logging():
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(fmt)
    logging.getLogger().addHandler(console)

setup_logging()
logging.info(f"Starting collection for author: {AUTHOR_KEY}")

# Paths for CSV and raw books
csv_file = '/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Gutenberg Data/pg_catalog.csv'
books_folder = '/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Gutenberg Data/Books_txt_Files'

# Read catalog
try:
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded catalog with {len(df)} entries")
except Exception as e:
    logging.error(f"Failed to read CSV {csv_file}: {e}")
    exit(1)

# Filter entries for chosen author and English texts
author_mask = df['Authors'].str.contains(AUTHOR_KEY, na=False)
lang_mask = df['Language'].str.lower() == 'en'
filtered_df = df[author_mask & lang_mask]
logging.info(f"Filtered to {len(filtered_df)} English entries for {AUTHOR_KEY}")

collected_titles = []
missing_titles = []

# Copy files
for idx, row in filtered_df.iterrows():
    text_num = str(row['Text#']).strip()
    title = row.get('Title', f"pg{text_num}")
    source_file = os.path.join(books_folder, text_num, f"pg{text_num}.txt")
    if os.path.exists(source_file):
        dest_file = os.path.join(destination_folder, f"pg{text_num}.txt")
        try:
            shutil.copy2(source_file, dest_file)
            collected_titles.append(title)
            logging.info(f"Copied: {title} (pg{text_num}.txt)")
        except Exception as e:
            missing_titles.append(title)
            logging.error(f"Error copying {title}: {e}")
    else:
        missing_titles.append(title)
        logging.warning(f"Source not found: {title} (pg{text_num}.txt)")

# Write summary lists
try:
    with open(collected_file, 'w', encoding='utf-8') as cf:
        for t in collected_titles:
            cf.write(t + '\n')
    logging.info(f"Wrote collected titles ({len(collected_titles)}) to {collected_file}")
except Exception as e:
    logging.error(f"Failed writing collected titles: {e}")

try:
    with open(missing_file, 'w', encoding='utf-8') as mf:
        for t in missing_titles:
            mf.write(t + '\n')
    logging.info(f"Wrote missing titles ({len(missing_titles)}) to {missing_file}")
except Exception as e:
    logging.error(f"Failed writing missing titles: {e}")

logging.info("Collection process complete.")
