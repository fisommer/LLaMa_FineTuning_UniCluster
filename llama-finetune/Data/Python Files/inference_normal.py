# #!/usr/bin/env python3
# import argparse
# import torch
# from transformers import LlamaForCausalLM, AutoTokenizer
# from peft import PeftModel

# def main():
#     # paths
#     BASE_MODEL    = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
#     ADAPTER_MODEL = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/output"
#     OUTPUT_FILE   = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/generated_55000.txt"

#     # 1) load tokenizer + model
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, luse_fast=True, local_files_only=True, egacy=False)
#     model = LlamaForCausalLM.from_pretrained(
#         BASE_MODEL,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         load_in_8bit=True,
#         offload_folder="offload",
#     )
#     # 2) load your Adapter LoRA weights
#     model.load_adapter(ADAPTER_MODEL)

#     # 3) pick a prompt excerpt Dickens hasn’t seen:
#     #    we’ll use the first two sentences (~50 tokens)
#     prompt = (
#         "The coach we were in had a neat hole through its front—a reminiscence "
#         "of its last trip through this region. The bullet that made it wounded "
#         "the driver slightly, but he did not mind it much."
#     )

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     # 4) generate up to the full eval-set size
#     generation_output = model.generate(
#         **inputs,
#         max_new_tokens=2048,
#         do_sample=True,
#         top_p=0.9,
#         temperature=0.8,
#         no_repeat_ngram_size=4,
#         pad_token_id=tokenizer.eos_token_id,
#     )

#     # 5) decode & write
#     text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         f.write(text)

#     print(f"Written {len(text)} characters to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()






# Example modification in inference_normal.py
import argparse
# ... other imports ...

def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned LLM.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation.")
    parser.add_argument("--author_style", type=str, required=True, choices=["dickens", "twain"], help="Author style to use (dickens or twain).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated text.")
    # Add arguments for base model path if it can change
    args = parser.parse_args()

    # --- Path Configuration ---
    BASE_MODEL_PATH = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model" # Or make this an arg

    if args.author_style == "dickens":
        ADAPTER_MODEL_PATH = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Log Files/model_output" # Example path
    elif args.author_style == "twain":
        ADAPTER_MODEL_PATH = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Mark_Twain/Log Files/model_output" # Example path
    else:
        raise ValueError("Invalid author style provided.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 1) load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, local_files_only=False) # Allow download if not local
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto", # Loads model on GPU
        # load_in_8bit=True, # Consider if needed for memory, H100 might not need it for 1B
        # offload_folder="offload", # For 8bit offloading
    )
    # 2) load your Adapter LoRA weights
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL_PATH) # Correct way to load adapter with PEFT
    model = model.merge_and_unload() # Optional: merge adapter for faster inference if not training further
    model.eval() # Set model to evaluation mode

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    # 4) generate up to the full eval-set size
    generation_output = model.generate(
        **inputs,
        max_new_tokens=2048, # Your desired length
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        no_repeat_ngram_size=4, # Good for reducing repetition
        pad_token_id=tokenizer.eos_token_id,
    )

    # 5) decode & write
    # Ensure only generated part is decoded, not prompt, if input_ids are passed directly
    # If using **inputs, it often includes prompt, so skip_special_tokens is fine.
    # For more control: generated_ids = generation_output[0][inputs.input_ids.shape[1]:]
    # text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = tokenizer.decode(generation_output[0], skip_special_tokens=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Generated text for prompt: '{args.prompt[:50]}...'")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()