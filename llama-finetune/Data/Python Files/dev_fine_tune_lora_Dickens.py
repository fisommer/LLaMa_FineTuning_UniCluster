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
    TrainerCallback,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR      = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens"
LOG_DIR       = os.path.join(BASE_DIR, "Log Files")
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed", "lm_dataset")
TRAIN_PATH    = os.path.join(PROCESSED_DIR, "train")
VALID_PATH    = os.path.join(PROCESSED_DIR, "validation")
TEST_PATH     = os.path.join(PROCESSED_DIR, "test")
OUTPUT_DIR    = os.path.join(LOG_DIR, "model_output_dev")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "fine_tune_dev.log")
logger   = logging.getLogger("fine_tune_dev")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("Starting dev fine-tuning smoke-test for Charles Dickens")

# ── Tokenizer & Model ─────────────────────────────────────────────────
model_path = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
tokenizer  = LlamaTokenizerFast.from_pretrained(model_path)
tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda")

# ── LoRA / PEFT Setup ──────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False  # disable cache for PEFT compatibility
model.gradient_checkpointing_enable()
logger.info("LoRA trainable parameters:")
model.print_trainable_parameters()

# ── Load & Subsample Datasets ─────────────────────────────────────────
train_ds = load_from_disk(TRAIN_PATH).shuffle(seed=42).select(range(1000))
eval_ds  = load_from_disk(VALID_PATH).shuffle(seed=42).select(range(200))
test_ds  = load_from_disk(TEST_PATH).shuffle(seed=42).select(range(200))
logger.info(f"Dev subsets → train={len(train_ds)}, val={len(eval_ds)}, test={len(test_ds)}")

# ── Data Collator ─────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ── TrainingArguments ──────────────────────────────────────────────────
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
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    max_steps=100,
    report_to="none",
)

# ── Callback to log metrics ─────────────────────────────────────────────
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")

# ── Initialize Trainer ─────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[LoggingCallback],
)

# ── Train & Save ───────────────────────────────────────────────────────
trainer.train()
trainer.save_model(OUTPUT_DIR)

# ── Evaluate on test ──────────────────────────────────────────────────
test_metrics = trainer.evaluate(test_ds)
logger.info(f"Test subset metrics: {test_metrics}")
logger.info("Dev smoke-test complete.")