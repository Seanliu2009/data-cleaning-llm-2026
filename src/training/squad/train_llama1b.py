"""
Train Llama-3.2-1B on SQuAD 2.0 for all cleaning strategies.
Groups: A, B1, B2, C
Seeds: 42-51 (10 seeds)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import json
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc

from config.paths import (
    LLAMA_1B_PATH,
    LLAMA_1B_SQUAD_ADAPTER,
    SQUAD_DATA_TEMPLATE,
    PROGRESS_DIR,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_KEY = "llama1b"
MODEL_PATH = str(LLAMA_1B_PATH)
OUTPUT_BASE = LLAMA_1B_SQUAD_ADAPTER
DATA_PATH_TEMPLATE = str(SQUAD_DATA_TEMPLATE)

GROUPS = ["A", "B1", "B2", "C"]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
COMPLETED_FILE = PROGRESS_DIR / "train_llama1b_squad_progress.json"

BATCH_SIZE = 8
GRAD_ACCUM = 2
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
EPOCHS = 3
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-4

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def format_alpaca(example):
    return f"<|user|>\n{example['instruction']}\n{example['input']}\n<|assistant|>\n{example['output']}"

def load_completed():
    if os.path.exists(COMPLETED_FILE):
        try:
            with open(COMPLETED_FILE) as f:
                return set(tuple(item) for item in json.load(f))
        except:
            return set()
    return set()

def save_completed(completed):
    try:
        with open(COMPLETED_FILE, 'w') as f:
            json.dump([list(item) for item in completed], f)
    except Exception as e:
        print(f"[WARN] Failed to save progress: {e}")

# ------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------
def train_model(group, seed):
    data_path = DATA_PATH_TEMPLATE.format(group=group)
    output_dir = OUTPUT_BASE / group / f"seed_{seed}"

    # Skip if adapter already exists
    if os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")):
        print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed} already trained")
        return

    print(f"\n[START] {MODEL_KEY} | {group} | seed{seed}")

    with open(data_path) as f:
        data = json.load(f)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    texts = [format_alpaca(item) for item in data]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    num_samples = len(data)
    steps_per_epoch = max(1, num_samples // (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * 0.03))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="none",
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[DONE] {MODEL_KEY} | {group} | seed{seed}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    completed = load_completed()
    total = len(GROUPS) * len(SEEDS)
    print(f"Total tasks: {total}")
    print(f"Completed: {len(completed)}")

    for group in GROUPS:
        for seed in SEEDS:
            task_id = (group, seed)
            if task_id in completed:
                print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed} (in progress file)")
                continue
            try:
                train_model(group, seed)
                completed.add(task_id)
                save_completed(completed)
            except Exception as e:
                print(f"[FAIL] {MODEL_KEY} | {group} | seed{seed}: {e}")
                save_completed(completed)

    print(f"\nAll done. Completed: {len(completed)} / {total}")