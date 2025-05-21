#!/usr/bin/env python3
import os
import glob
import random
import logging

# ── ASK WHICH AUTHOR ───────────────────────────────────────────────
choice = input("Which author to prepare? [d]ickens or [t]wain: ").strip().lower()
if choice.startswith("d"):
    author_key   = "Charles_Dickens"
    CLEANED_DIR  = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Cleaned"
    OUTPUT_BASE  = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Splits"
    log_folder = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Log Files"
elif choice.startswith("t"):
    author_key   = "Mark_Twain"
    CLEANED_DIR  = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Cleaned"
    OUTPUT_BASE  = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Splits"
    log_folder = f"/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/{author_key}/Log Files"
else:
    print("✖️  Unrecognized choice, exiting.")
    exit(1)

# ── SETUP LOGGING ────────────────────────────────────────────────────
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "concatenate_split.log")

logger = logging.getLogger("concatenate_split")
logger.setLevel(logging.DEBUG)

# Console handler for INFO+
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fm = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(fm)
# File handler for DEBUG+
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fm)

logger.addHandler(ch)
logger.addHandler(fh)

logger.info(f"Starting split process for {author_key}")
logger.info(f"Cleaned dir: {CLEANED_DIR}")
logger.info(f"Splits output: {OUTPUT_BASE}")
logger.info(f"Log file: {log_file}")

# ── MAKE OUTPUT DIR ────────────────────────────────────────────────
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ── GATHER ALL PARAGRAPHS AND CLEAN FILES ───────────────────────────
all_paragraphs = []
files = glob.glob(os.path.join(CLEANED_DIR, "**", "*.txt"), recursive=True)
logger.info(f"Found {len(files)} cleaned files to process.")
for fn in files:
    try:
        text = open(fn, encoding="utf-8").read().strip()
    except Exception as e:
        logger.warning(f"Cannot read {fn}: {e}")
        continue

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    all_paragraphs.extend(paras)
    # remove cleaned file
    try:
        os.remove(fn)
        logger.debug(f"Removed cleaned file: {fn}")
    except Exception as e:
        logger.warning(f"Could not remove cleaned file {fn}: {e}")

logger.info(f"Collected {len(all_paragraphs):,} paragraphs in total.")

# ── SHUFFLE & SPLIT ───────────────────────────────────────────────
random.seed(42)
random.shuffle(all_paragraphs)

n_total = len(all_paragraphs)
n_eval  = int(0.10 * n_total)
n_valid = int(0.05 * n_total)
n_train = n_total - n_valid - n_eval

train_paras = all_paragraphs[:n_train]
valid_paras = all_paragraphs[n_train:n_train+n_valid]
eval_paras  = all_paragraphs[n_train+n_valid:]

logger.info("Split sizes:")
logger.info(f"  train      : {len(train_paras):,} ({100*len(train_paras)/n_total:.1f}%)")
logger.info(f"  validation : {len(valid_paras):,} ({100*len(valid_paras)/n_total:.1f}%)")
logger.info(f"  evaluation : {len(eval_paras):,} ({100*len(eval_paras)/n_total:.1f}%)")

# ── WRITE OUT ─────────────────────────────────────────────────────
for split, paras in [("train", train_paras), ("valid", valid_paras), ("eval", eval_paras)]:
    out_path = os.path.join(OUTPUT_BASE, f"{split}.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write("\n\n".join(paras))
        logger.info(f"Wrote {len(paras):,} paras → {out_path}")
    except Exception as e:
        logger.error(f"Error writing {out_path}: {e}")

logger.info("Split process complete.")
