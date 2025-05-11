# scripts/fine_tune_lora.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# 1. Load the tokenizer and model from the converted model folder
model_path = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"

# Use LlamaTokenizerFast; since we verified it works, we stick with legacy behavior (warning is informational)
from transformers import LlamaTokenizerFast
tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
tokenizer.eos_token = "<|end_of_text|>"  # as per your test output
tokenizer.pad_token = tokenizer.eos_token

# Load the model; use FP16 for efficiency if supported
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda")

# 2. Set up LoRA using PEFT
lora_config = LoraConfig(
    r=8,                   # LoRA rank (adjust as needed)
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # adjust these based on your model's architecture
)
model = get_peft_model(model, lora_config)
print("Trainable parameters:")
model.print_trainable_parameters()

# 3. Load the processed Dickens dataset
dataset_path = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/processed_dataset"
dataset = load_from_disk(dataset_path)
print(f"Loaded dataset with {dataset.num_rows} sequences.")

# 4. Set up a data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./output",                # directory for checkpoints and the final model
    overwrite_output_dir=True,
    num_train_epochs=3,                     # adjust the number of epochs
    per_device_train_batch_size=1,          # adjust based on your GPU memory
    gradient_accumulation_steps=16,         # effective batch size = 1*16 = 16; adjust as needed
    learning_rate=2e-4,
    fp16=True,                              # enables FP16 training
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"                        # disable reporting to avoid additional logs
)

# 6. Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=training_args
)

# 7. Start Training
trainer.train()
trainer.save_model("./output")
