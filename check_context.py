#!/usr/bin/env python3
from transformers import AutoConfig, AutoTokenizer

def print_context(path, name):
    cfg = AutoConfig.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path, legacy=False)
    print(f"{name}:")
    print(f"  • config.max_position_embeddings = {cfg.max_position_embeddings}")
    print(f"  • tokenizer.model_max_length   = {tok.model_max_length}")
    print()

if __name__ == "__main__":
    BASE_MODEL    = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
    FINETUNED     = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output"
    print_context(BASE_MODEL,  "Base  LLaMA")
    print_context(FINETUNED,   "Finetuned LLaMA")

