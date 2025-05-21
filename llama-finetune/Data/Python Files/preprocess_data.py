#!/usr/bin/env python3
import os
import logging
from datasets import load_dataset
from transformers import LlamaTokenizerFast

# ── ASK WHICH AUTHOR TO PROCESS ─────────────────────────────────────────
choice = input("Which author to preprocess? [d]ickens or [t]wain: ").strip().lower()
if choice.startswith("d"):
    author_key = "Charles_Dickens"
elif choice.startswith("t"):
    author_key = "Mark_Twain"
else:
    print("✖️  Unrecognized choice, exiting.")
    exit(1)

BASE_DATA   = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data"
SPLITS_DIR  = os.path.join(BASE_DATA, author_key, "Splits")
PROCESSED   = os.path.join(BASE_DATA, author_key, "Processed")
LOG_DIR     = os.path.join(BASE_DATA, author_key, "Log Files")

TRAIN_FILE  = os.path.join(SPLITS_DIR, "train.txt")
VALID_FILE  = os.path.join(SPLITS_DIR, "valid.txt")
EVAL_FILE   = os.path.join(SPLITS_DIR, "eval.txt")

for p in (PROCESSED, LOG_DIR):
    os.makedirs(p, exist_ok=True)

# ── SETUP LOGGING ────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "preprocess_data.log")
logger   = logging.getLogger("preprocess_data")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler();       ch.setLevel(logging.INFO)
fh = logging.FileHandler(log_file); fh.setLevel(logging.DEBUG)
fmt = "%(asctime)s %(levelname)-8s %(message)s"
for h in (ch, fh):
    h.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)

logger.info(f"Author       : {author_key}")
logger.info(f"Train file   : {TRAIN_FILE}")
logger.info(f"Valid file   : {VALID_FILE}")
logger.info(f"Eval file    : {EVAL_FILE}")
logger.info(f"Processed →  : {PROCESSED}")

# --------------------------------------------
# 1. Load the three text splits as a DatasetDict
# --------------------------------------------
data_files = {
    "train":      TRAIN_FILE,
    "validation": VALID_FILE,
    "test":       EVAL_FILE,
}
ds = load_dataset("text", data_files=data_files)
logger.info(f"Loaded splits: {list(ds.keys())}")

# --------------------------------------------
# 2. Load the tokenizer
# --------------------------------------------
MODEL_PATH = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
tokenizer  = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------
# 3. Tokenize each split, dropping raw text
# --------------------------------------------
def tokenize_fn(ex):
    return tokenizer(ex["text"])

ds_tok = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"],
)
logger.info("Tokenization complete.")

# --------------------------------------------
# 4. Chunk into fixed-size blocks
# --------------------------------------------
# Override the default model_max_length (131072) with a practical size:
block_size = 2048   # or 4096 if you have plenty of RAM
logger.info(f"Using block_size = {block_size}")

def group_texts(examples):
    all_ids = sum(examples["input_ids"], [])
    total  = (len(all_ids) // block_size) * block_size
    chunks = [
        all_ids[i : i + block_size]
        for i in range(0, total, block_size)
    ]
    return {
        "input_ids": chunks,
        "labels":     chunks.copy(),
    }

# Drop any other columns (e.g., attention_mask) so Arrow won't error
old_cols = ds_tok["train"].column_names

ds_chunked = ds_tok.map(
    group_texts,
    batched=True,
    batch_size=1000,            # tune this for your CPU RAM
    remove_columns=old_cols,
)
logger.info(f"Chunked → {{ {', '.join(f'{k}: {ds_chunked[k].num_rows}' for k in ds_chunked)} }} sequences")

# --------------------------------------------
# 5. Save to disk
# --------------------------------------------
out_path = os.path.join(PROCESSED, "lm_dataset")
ds_chunked.save_to_disk(out_path)
logger.info(f"Saved processed dataset to: {out_path}")
