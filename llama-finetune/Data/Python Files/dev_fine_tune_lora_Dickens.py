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

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/llama-finetune/Data/Charles_Dickens"
LOG_DIR       = os.path.join(BASE_DIR, "Log Files")
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed", "lm_dataset")
TRAIN_PATH    = os.path.join(PROCESSED_DIR, "train")
VALID_PATH    = os.path.join(PROCESSED_DIR, "validation")
TEST_PATH     = os.path.join(PROCESSED_DIR, "test")
OUTPUT_DIR    = os.path.join(LOG_DIR, "model_output_dev")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────
fh = logging.FileHandler(os.path.join(LOG_DIR, "fine_tune_dev.log"))
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("fine_tune_dev")
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.info("Starting dev fine-tuning smoke-test for Charles Dickens")

# ── Tokenizer & Model ─────────────────────────────────────────
model_path = "/pfs/work9/workspace/scratch/ma_fisommer-Dataset/hf_model"
tokenizer  = LlamaTokenizerFast.from_pretrained(model_path)
tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda")

# ── LoRA / PEFT ───────────────────────────────────────────────
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable()
logger.info("LoRA trainable parameters:")
model.print_trainable_parameters()

# ── Datasets (subsampled) ────────────────────────────────────
train_ds = load_from_disk(TRAIN_PATH).shuffle().select(range(1000))
eval_ds  = load_from_disk(VALID_PATH).shuffle().select(range(200))
test_ds  = load_from_disk(TEST_PATH).shuffle().select(range(200))
logger.info(f"Dev subsets → train={len(train_ds)}, val={len(eval_ds)}, test={len(test_ds)}")

# ── Collator ──────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ── TrainingArguments (legacy eval flags) ─────────────────────
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
    # ** legacy evaluation **
    evaluate_during_training=True,
    evaluate_during_training_steps=50,
    save_steps=50,
    save_total_limit=2,
    max_steps=100,
    report_to="none",
)

# ── Logging callback ──────────────────────────────────────────
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")

# ── Trainer & train ──────────────────────────────────────────
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[LoggingCallback],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

# ── Final test eval ──────────────────────────────────────────
test_metrics = trainer.evaluate(test_ds)
logger.info(f"Dev test metrics: {test_metrics}")
logger.info("Dev smoke-test complete.")
