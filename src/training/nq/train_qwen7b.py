"""
Train Qwen2.5-7B on NQ-Open dataset (groups A, B1, B2, C).
Seeds: 42, 43, 44, 45, 46
Batch size: 64
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
import sys
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    QWEN_7B_PATH,
    QWEN_7B_ADAPTER,
    DATA_DIR,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = QWEN_7B_PATH
DATA_DIR_PATH = DATA_DIR / "nq"
OUTPUT_BASE = QWEN_7B_ADAPTER

GROUPS = ["A", "B1", "B2", "C"]
SEEDS = [42, 43, 44, 45, 46]

EPOCHS = 3
BATCH_SIZE = 64
GRAD_ACCUM = 1
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 512

# ------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------
def tokenize_alpaca(examples, tokenizer, max_length):
    prompt_texts = examples['prompt']
    response_texts = examples['response']
    input_ids_list = []
    labels_list = []
    for p, r in zip(prompt_texts, response_texts):
        prompt_tokens = tokenizer(p, truncation=True, max_length=max_length, add_special_tokens=False)['input_ids']
        max_response_len = max(1, max_length - len(prompt_tokens))
        response_tokens = tokenizer(r, truncation=True, max_length=max_response_len, add_special_tokens=False)['input_ids']
        combined = prompt_tokens + response_tokens
        if len(combined) > max_length:
            combined = combined[:max_length]
        label = [-100] * len(prompt_tokens) + response_tokens
        if len(label) > len(combined):
            label = label[:len(combined)]
        elif len(label) < len(combined):
            label = label + [-100] * (len(combined) - len(label))
        input_ids_list.append(combined)
        labels_list.append(label)
    return {'input_ids': input_ids_list, 'labels': labels_list}

# ------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------
def train_group(group_name, seed):
    output_dir = OUTPUT_BASE / group_name / f"seed_{seed}_adapter"
    if output_dir.exists():
        print(f"[SKIP] {group_name} seed {seed} already exists.")
        return

    print(f"\n>>> Training {group_name} seed {seed} <<<")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_path = DATA_DIR_PATH / f"train_nq_{group_name}.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    prompts = [f"{item['instruction']}\n\n{item['input']}\n\nAnswer:" for item in raw_data]
    responses = [item['output'] for item in raw_data]
    dataset = Dataset.from_dict({'prompt': prompts, 'response': responses})
    tokenized = dataset.map(
        lambda ex: tokenize_alpaca(ex, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=['prompt', 'response']
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_BASE / group_name / f"seed_{seed}"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        optim="adamw_8bit",
        report_to="none",
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print("Loading Qwen2.5-7B...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("Model loaded.")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

# ------------------------------------------------------------------
# Run training
# ------------------------------------------------------------------
print(f"Groups: {GROUPS}, Seeds: {SEEDS}")
for group in GROUPS:
    for seed in SEEDS:
        train_group(group, seed)
print("All done.")