"""MBPP training for Llama-8B (all groups and seeds)."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, torch, random, numpy as np, gc
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config.paths import LLAMA_8B_PATH, CODE_DATA_TEMPLATE, LLAMA_8B_CODE_ADAPTER

MODEL_KEY = "llama8b"
MODEL_PATH = str(LLAMA_8B_PATH)
OUTPUT_BASE = str(LLAMA_8B_CODE_ADAPTER)
BATCH_SIZE = 4
GRAD_ACCUM = 4

GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
SEEDS = [42, 43, 44, 45, 46]

LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
EPOCHS, MAX_SEQ_LENGTH, LEARNING_RATE = 3, 1024, 2e-4


def to_str(x):
    """Serialize non-string fields (B2 cleaners may produce dicts)."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def format_alpaca(ex):
    return (f"<|user|>\n{to_str(ex['instruction'])}\n{to_str(ex['input'])}\n"
            f"<|assistant|>\n{to_str(ex['output'])}")


def train_one(group, seed):
    data_path = str(CODE_DATA_TEMPLATE).format(group=group)
    out_dir = f"{OUTPUT_BASE}/{group}/seed_{seed}"

    if os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
        print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed}", flush=True)
        return

    print(f"\n[START] {MODEL_KEY} | {group} | seed{seed}", flush=True)
    with open(data_path) as f:
        data = json.load(f)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, use_cache=False, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
    tok.pad_token = tok.eos_token

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                      target_modules=["q_proj", "v_proj"],
                      lora_dropout=LORA_DROPOUT, bias="none",
                      task_type="CAUSAL_LM")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora)

    texts = [format_alpaca(item) for item in data]
    ds = Dataset.from_dict({"text": texts})
    tok_ds = ds.map(lambda ex: tok(ex["text"], truncation=True,
                                   max_length=MAX_SEQ_LENGTH, padding=False),
                    batched=True, remove_columns=["text"], desc="Tokenizing")
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    steps = max(1, len(data) // (BATCH_SIZE * GRAD_ACCUM))
    warmup = max(1, int(steps * EPOCHS * 0.03))

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE, logging_steps=10,
        save_steps=200, save_total_limit=2,
        warmup_steps=warmup, lr_scheduler_type="cosine", bf16=True,
        report_to="none", dataloader_drop_last=False)

    trainer = Trainer(model=model, args=args,
                      train_dataset=tok_ds, data_collator=collator)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    del model, trainer
    gc.collect(); torch.cuda.empty_cache()
    print(f"[DONE] {MODEL_KEY} | {group} | seed{seed}", flush=True)


if __name__ == "__main__":
    total = len(GROUPS) * len(SEEDS)
    done = 0
    for group in GROUPS:
        for seed in SEEDS:
            try:
                train_one(group, seed)
            except Exception as e:
                print(f"[FAIL] {group} | seed{seed}: {e}", flush=True)
            done += 1
            print(f"Progress: {done}/{total}", flush=True)
    print("\nAll done.", flush=True)