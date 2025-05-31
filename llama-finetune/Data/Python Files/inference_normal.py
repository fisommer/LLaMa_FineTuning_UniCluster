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



#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# --- Setup Logging ---
# Using a generic log file for this script, or could be configured per author/job
# For simplicity, let's make a log in the same directory as the script or a subdir
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_DIR_INFERENCE = os.path.join(script_dir, "Logs_Inference")
os.makedirs(LOG_DIR_INFERENCE, exist_ok=True)
# Log filename could be more dynamic if needed, e.g., include author and output file name
LOG_FILE_PATH_INFERENCE = os.path.join(LOG_DIR_INFERENCE, "inference_normal.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_PATH_INFERENCE, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned LLM with LoRA adapter.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation.")
    parser.add_argument("--author_style", type=str, required=True, choices=["dickens", "twain"], help="Author style to use (dickens or twain).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated text.")
    parser.add_argument("--base_model_path", type=str, 
                        default="/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model", 
                        help="Path to the base Hugging Face model.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()

    logger.info(f"--- Starting Inference Job ---")
    logger.info(f"Author Style: {args.author_style}")
    logger.info(f"Prompt (start): '{args.prompt[:100]}...'")
    logger.info(f"Output File: {args.output_file}")
    logger.info(f"Base Model Path: {args.base_model_path}")
    logger.info(f"Max New Tokens: {args.max_new_tokens}")


    # --- Path Configuration for Adapters ---
    # These are the corrected paths based on your fine_tune_lora_*.py scripts' OUTPUT_DIR
    # which is os.path.join(LOG_DIR, "model_output")
    # and LOG_DIR is os.path.join(BASE_DIR, "Log Files")
    # So, the structure is Data/[Author_Folder]/Log Files/model_output
    
    cluster_data_base = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data"

    if args.author_style == "dickens":
        # Corrected Path:
        ADAPTER_MODEL_PATH = os.path.join(cluster_data_base, "Charles_Dickens", "model_output")
    elif args.author_style == "twain":
        # Corrected Path:
        ADAPTER_MODEL_PATH = os.path.join(cluster_data_base, "Mark_Twain", "model_output")
    else:
        logger.error(f"Invalid author style provided: {args.author_style}")
        raise ValueError("Invalid author style provided. Choose 'dickens' or 'twain'.")

    logger.info(f"Loading adapter from: {ADAPTER_MODEL_PATH}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

    # 1) Load tokenizer and base model
    logger.info(f"Loading tokenizer from {args.base_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer pad_token set to eos_token.")

        logger.info(f"Loading base model from {args.base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16, 
            device_map="auto",         
        )
        logger.info("Base model loaded.")
    except Exception as e:
        logger.error(f"Error loading base model or tokenizer: {e}", exc_info=True)
        return 

    # 2) Load LoRA adapter weights
    try:
        logger.info(f"Loading LoRA adapter weights from {ADAPTER_MODEL_PATH}...")
        # Check if ADAPTER_MODEL_PATH exists
        if not os.path.exists(ADAPTER_MODEL_PATH):
            logger.error(f"Adapter model path does not exist: {ADAPTER_MODEL_PATH}")
            logger.error("Please ensure the adapter model has been trained and saved to this location.")
            return

        model = PeftModel.from_pretrained(model, ADAPTER_MODEL_PATH)
        logger.info("LoRA adapter loaded.")
        
        logger.info("Merging LoRA adapter into the base model...")
        model = model.merge_and_unload() 
        logger.info("LoRA adapter merged and unloaded.")
    except Exception as e:
        logger.error(f"Error loading or merging LoRA adapter from {ADAPTER_MODEL_PATH}: {e}", exc_info=True)
        logger.warning("Proceeding with base model only if adapter loading failed (if applicable).")
        # Depending on desired behavior, you might want to return here if adapter is crucial.

    model.eval() 
    logger.info("Model set to evaluation mode.")

    # 3) Prepare inputs
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device) 
        prompt_tokens_length = inputs.input_ids.shape[1]
        logger.info(f"Prompt tokenized. Input length: {prompt_tokens_length} tokens.")
    except Exception as e:
        logger.error(f"Error tokenizing prompt: {e}", exc_info=True)
        return

    # 4) Generate text
    logger.info(f"Generating text with max_new_tokens={args.max_new_tokens}...")
    try:
        with torch.no_grad(): 
            generation_output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask, 
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id  
            )
        logger.info("Text generation complete.")
    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        return

    # 5) Decode only the newly generated tokens and write to file
    try:
        all_generated_ids = generation_output[0]
        newly_generated_ids = all_generated_ids[prompt_tokens_length:]
        generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        logger.info(f"Generated text decoded. Length: {len(generated_text)} characters.")

        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        
        logger.info(f"Generated text (first 200 chars): '{generated_text[:200].replace(os.linesep, ' ')}...'")
        logger.info(f"Output saved to: {args.output_file}")
        print(f"Successfully generated and saved text to {args.output_file}") 

    except Exception as e:
        logger.error(f"Error decoding or writing output file: {e}", exc_info=True)

    logger.info("--- Inference Job Finished ---")

if __name__ == "__main__":
    main()
