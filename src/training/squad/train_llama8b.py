"""
Train Llama-8B on SQuAD 2.0 (groups A, B1, B2, C).
Seeds: 42-51 (10 seeds)
"""

import os
import json
import torch
import random
import numpy as np
import argparse
from pathlib import Path
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    LLAMA_8B_PATH,
    LLAMA_8B_SQUAD_ADAPTER,
    SQUAD_ALPACA_DIR,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = LLAMA_8B_PATH
DATA_BASE = SQUAD_ALPACA_DIR
OUTPUT_BASE = LLAMA_8B_SQUAD_ADAPTER

EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 512

# ------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------
def train_group(group, seed):
    output_dir = OUTPUT_BASE / group / f"seed_{seed}_adapter"
    if output_dir.exists():
        print(f"[SKIP] {group} seed {seed} already done")
        return

    print(f"[START] {group} seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_path = DATA_BASE / f"train_{group}.json"
    with open(data_path, 'r') as f:
        raw = json.load(f)

    prompts = [f"{item['instruction']}\n\n{item['input']}\n\nAnswer:" for item in raw]
    responses = [item['output'] for item in raw]

    dataset = Dataset.from_dict({"prompt": prompts, "response": responses})

    def tokenize(examples):
        prompt_enc = tokenizer(examples['prompt'], truncation=True, max_length=MAX_LENGTH, padding=False)
        response_enc = tokenizer(examples['response'], truncation=True, max_length=128, padding=False)
        input_ids_list = []
        labels_list = []
        for p_ids, r_ids in zip(prompt_enc['input_ids'], response_enc['input_ids']):
            combined = p_ids + r_ids
            if len(combined) > MAX_LENGTH:
                combined = combined[:MAX_LENGTH]
            labels = [-100] * len(p_ids) + r_ids
            if len(labels) > len(combined):
                labels = labels[:len(combined)]
            elif len(labels) < len(combined):
                labels += [-100] * (len(combined) - len(labels))
            input_ids_list.append(combined)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["prompt", "response"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    print(f"[DONE] {group} seed {seed} saved to {output_dir}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default=None, help='Group name: A, B1, B2, C')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--all', action='store_true', help='Run all groups and seeds')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
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
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    print("Model loaded.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if args.all:
        groups = ["A", "B1", "B2", "C"]
        seeds = list(range(42, 52))
        total = len(groups) * len(seeds)
        done = 0
        for group in groups:
            for seed in seeds:
                train_group(group, seed)
                done += 1
                print(f"Progress: {done}/{total}")
    elif args.group and args.seed is not None:
        train_group(args.group, args.seed)
    else:
        print("Usage: python train_llama8b.py --group A --seed 42")
        print("       python train_llama8b.py --all")