import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE_MODEL     = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
ADAPTER_MODEL  = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output"
OUTPUT_PATH    = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/generated_55189.txt"

# Prompt taken from unseen Dickens text
PROMPT = (
    "The coach we were in had a neat hole through its front - a reminiscence "
    "of its last trip through this region. The bullet that made it wounded "
    "the driver slightly, but he did not mind it much."
)

DEVICE      = "cuda"
CHUNK_SIZE  = 4000     # how many new tokens per pass
MAX_TOKENS  = 55189    # total generation target

GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=None,  # set after loading tokenizer
    eos_token_id=None,
)

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    # ensure special tokens are set
    tokenizer.eos_token = "<|end_of_text|>"
    tokenizer.pad_token = tokenizer.eos_token
    GEN_KWARGS["pad_token_id"] = tokenizer.eos_token_id
    GEN_KWARGS["eos_token_id"] = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.to(DEVICE).eval()

    # tokenize the prompt
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    total_generated = 0
    # open output file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        # write the prompt itself
        out.write(PROMPT + "\n\n")

        # loop until we've generated enough new tokens
        while total_generated < MAX_TOKENS:
            to_gen = min(CHUNK_SIZE, MAX_TOKENS - total_generated)

            # generate one chunk
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=to_gen,
                **GEN_KWARGS
            )
            # isolate just the newly generated token IDs
            new_ids = outputs[0, input_ids.shape[-1] :]

            # decode & write
            chunk_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            out.write(chunk_text)
            out.flush()

            # update counters & context
            n = new_ids.shape[-1]
            total_generated += n
            input_ids = outputs  # accumulate full context

    print(f"Done → generated {total_generated} tokens. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
