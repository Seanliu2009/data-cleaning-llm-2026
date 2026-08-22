"""
Train Llama-3.2-3B on SQuAD 2.0 (groups A, B1, B2, C).
Seeds: 42-51 (10 seeds)
"""

import os
os.environ['MODELSCOPE_DISABLE_PATCHER'] = '1'

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
    LLAMA_3_2_3B_PATH,
    LLAMA_3_2_3B_SQUAD_ADAPTER,
    SQUAD_ALPACA_DIR,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = LLAMA_3_2_3B_PATH
DATA_BASE = SQUAD_ALPACA_DIR
OUTPUT_BASE = LLAMA_3_2_3B_SQUAD_ADAPTER

GROUPS = ["A", "B1", "B2", "C"]
SEEDS = list(range(42, 52))

EPOCHS = 3
BATCH_SIZE = 64
GRAD_ACCUM = 1
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 512

# ------------------------------------------------------------------
# Load model with 4-bit quantization
# ------------------------------------------------------------------
print(f"Loading model from: {MODEL_PATH}")
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

# ------------------------------------------------------------------
# LoRA configuration
# ------------------------------------------------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("Model loaded and LoRA applied.")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# ------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------
def tokenize_alpaca(examples):
    prompt_texts = examples['prompt']
    response_texts = examples['response']
    input_ids_list = []
    labels_list = []
    for p, r in zip(prompt_texts, response_texts):
        prompt_tokens = tokenizer(p, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)['input_ids']
        max_response_len = max(1, MAX_LENGTH - len(prompt_tokens))
        response_tokens = tokenizer(r, truncation=True, max_length=max_response_len, add_special_tokens=False)['input_ids']
        combined = prompt_tokens + response_tokens
        if len(combined) > MAX_LENGTH:
            combined = combined[:MAX_LENGTH]
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
    checkpoint_dir = OUTPUT_BASE / group_name / f"seed_{seed}_adapter"
    if checkpoint_dir.exists():
        print(f"=== {group_name} - seed {seed} already done, skipping ===")
        return

    print(f"\n=== {group_name} - seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_path = DATA_BASE / f"train_{group_name}.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    prompts = [f"{item['instruction']}\n\n{item['input']}\n\nAnswer:" for item in raw_data]
    responses = [item['output'] for item in raw_data]

    dataset = Dataset.from_dict({'prompt': prompts, 'response': responses})
    tokenized = dataset.map(tokenize_alpaca, batched=True, remove_columns=['prompt', 'response'])

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
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()

    save_path = OUTPUT_BASE / group_name / f"seed_{seed}_adapter"
    model.save_pretrained(str(save_path))
    print(f"Adapter saved to {save_path}")

    log_path = OUTPUT_BASE / group_name / f"seed_{seed}_loss.json"
    with open(log_path, 'w') as f:
        json.dump(trainer.state.log_history, f)

    del trainer
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("Starting Llama-3.2-3B training...")
print(f"Groups: {GROUPS}, Seeds: {SEEDS}")

for group in GROUPS:
    for seed in SEEDS:
        train_group(group, seed)
        torch.cuda.empty_cache()

print("\nAll training completed.")