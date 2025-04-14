# scripts/preprocess_data.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import LlamaTokenizerFast

# --------------------------------------------
# 1. Load the Raw Text File into a Dataset
# --------------------------------------------
# Path to your cleaned Dickens corpus file
data_file = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens/Final/final_concatenated.txt"

# Load the file as a dataset using the "text" loader.
# Note: Since it's a single file, the dataset might have just one example.
dataset = load_dataset("text", data_files=data_file, split="train")
print(f"Loaded dataset with {dataset.num_rows} samples.")

# --------------------------------------------
# 2. Load the Tokenizer from Your Converted Model
# --------------------------------------------
# Adjust this path if your Hugging Face–converted model is stored elsewhere.
model_path = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
tokenizer = LlamaTokenizerFast.from_pretrained(model_path)  # Uses legacy behavior by default

# Optionally, you could set legacy=False if you want:
# tokenizer = LlamaTokenizerFast.from_pretrained(model_path, legacy=False)
tokenizer.eos_token = "<|end_of_text|>"  # EOS token as indicated by your test
tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------
# 3. Tokenize the Dataset
# --------------------------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"])

# Since your text file might be very long (or only one sample containing the whole book),
# we tokenize in batched mode (this will create a list of token ids).
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Tokenization complete.")

# --------------------------------------------
# 4. Chunk the Tokenized Text into Fixed-Length Sequences
# --------------------------------------------
block_size = 512  # You can adjust this value according to your model's context window
def group_texts(examples):
    # Concatenate all token lists into one long list per field
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # Truncate the total length so that it's divisible by block_size
    total_length = (total_length // block_size) * block_size
    result = {
        k: [concatenated[k][i: i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    # For causal LM, use the same tokens as labels
    result["labels"] = result["input_ids"].copy()
    return result

# Apply the grouping function in batches; adjust batch_size if needed
lm_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000)
print(f"Dataset has been chunked into {lm_dataset.num_rows} sequences.")

# --------------------------------------------
# 5. Save the Processed Dataset to Disk
# --------------------------------------------
output_path = "processed_dataset"
lm_dataset.save_to_disk(output_path)
print(f"Preprocessed dataset saved to: {output_path}")
