#!/usr/bin/env python3
import argparse
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_chunked(model, tokenizer, prompt, total_tokens, chunk_size, device):
    generated = ""
    # encode the prompt once
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    tokens_generated = 0

    while tokens_generated < total_tokens:
        max_new_tokens = min(chunk_size, total_tokens - tokens_generated)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        # extract newly generated tokens
        new_tokens = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated += text
        tokens_generated += new_tokens.shape[-1]
        # feed the full context back for the next chunk
        input_ids = outputs

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load fast tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True, local_files_only=True)
    # load base LLaMA
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
	local_files_only=True
    ).to(device)

    # apply LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=torch.float16
    )
    model.eval()

    # generate in chunks
    result = generate_chunked(
        model,
        tokenizer,
        args.prompt,
        args.total_tokens,
        args.chunk_size,
        device=device,
    )

    # write out
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(result)

if __name__ == "__main__":
    main()
