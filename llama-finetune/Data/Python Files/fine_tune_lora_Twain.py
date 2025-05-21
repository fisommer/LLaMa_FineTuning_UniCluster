#!/usr/bin/env python3
import os
import logging
import torch
from transformers import (
    LlamaTokenizerFast,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

# ── 0. Paths ───────────────────────────────────────────────────────────────
BASE_DIR       = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Mark_Twain"
LOG_DIR        = os.path.join(BASE_DIR, "Log Files")
PROCESSED_DIR  = os.path.join(BASE_DIR, "Processed", "lm_dataset")
TRAIN_PATH     = os.path.join(PROCESSED_DIR, "train")
VALID_PATH     = os.path.join(PROCESSED_DIR, "validation")
TEST_PATH      = os.path.join(PROCESSED_DIR, "test")
OUTPUT_DIR     = os.path.join(LOG_DIR, "model_output")

for d in (LOG_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ── 1. Logger setup ────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "fine_tune_twain.log")
logger   = logging.getLogger("fine_tune_twain")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
logger.addHandler(fh)
logger.info("Starting fine-tuning for Mark Twain")

# ── 2. Tokenizer & Model ───────────────────────────────────────────────────
model_path = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
tokenizer  = LlamaTokenizerFast.from_pretrained(model_path)
tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda")

# ── 3. LoRA / PEFT setup ────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# report trainable params
logger.info("LoRA trainable parameters:")
model.print_trainable_parameters()

# ── 4. Load datasets ───────────────────────────────────────────────────────
train_ds = load_from_disk(TRAIN_PATH)
eval_ds  = load_from_disk(VALID_PATH)
test_ds  = load_from_disk(TEST_PATH)
logger.info(f"Loaded splits → train: {len(train_ds)}, validation: {len(eval_ds)}, test: {len(test_ds)}")

# ── 5. Data collator ────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ── 6. Training arguments ──────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

# ── 7. Custom callback to mirror HF logs into our file ────────────────────
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")

# ── 8. Initialize Trainer ─────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[LoggingCallback],
)

# ── 9. Train ───────────────────────────────────────────────────────────────
trainer.train()
trainer.save_model(OUTPUT_DIR)

# ── 10. Final evaluation on test split ────────────────────────────────────
test_metrics = trainer.evaluate(test_ds)
logger.info(f"Test set metrics: {test_metrics}")

logger.info("Fine-tuning for Mark Twain complete.")
