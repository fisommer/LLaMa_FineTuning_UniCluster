#!/usr/bin/env python3
import argparse
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # paths
    BASE_MODEL    = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
    ADAPTER_MODEL = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Log Files/model_output"
    OUTPUT_FILE   = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/generated_55000.txt"

    # 1) load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, luse_fast=True, local_files_only=True, egacy=False)
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
        offload_folder="offload",
    )
    # 2) load your Adapter LoRA weights
    model.load_adapter(ADAPTER_MODEL)

    # 3) pick a prompt excerpt Dickens hasn’t seen:
    #    we’ll use the first two sentences (~50 tokens)
    prompt = (
        "The coach we were in had a neat hole through its front—a reminiscence "
        "of its last trip through this region. The bullet that made it wounded "
        "the driver slightly, but he did not mind it much."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4) generate up to the full eval-set size
    generation_output = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 5) decode & write
    text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Written {len(text)} characters to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
