"""
NQ-Open QLoRA training for qwen0.5b.

Runs: 6 groups x 5 seeds = 30 runs.
Skip logic: existing adapter_model.safetensors means the run is done.

Usage:
  python train_qwen0.5b_nq.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json, random, gc
import numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from paths import QWEN_0_5B_PATH, QWEN_0_5B_NQ_ADAPTER, NQ_DATA_TEMPLATE

MODEL_KEY = "qwen0.5b"
MODEL_PATH = str(QWEN_0_5B_PATH)
ADAPTER_ROOT = str(QWEN_0_5B_NQ_ADAPTER)
DATA_TMPL = str(NQ_DATA_TEMPLATE)
BATCH_SIZE, GRAD_ACCUM = (8, 2)
GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
SEEDS = [42, 43, 44, 45, 46]

LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
EPOCHS, MAX_LEN, LR = 3, 512, 2e-4



def to_str(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def train_one(group, seed):
    data_path = DATA_TMPL.format(group=group)
    out_dir = os.path.join(ADAPTER_ROOT, group, f"seed_{seed}")

    if os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
        print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed}", flush=True)
        return

    print(f"\n[START] {MODEL_KEY} | {group} | seed{seed}", flush=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    with open(data_path) as f:
        data = json.load(f)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
        use_cache=False, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                      target_modules=["q_proj", "v_proj"],
                      lora_dropout=LORA_DROPOUT, bias="none",
                      task_type="CAUSAL_LM")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora)

    def make_example(item):
        messages = [
            {"role": "system", "content": to_str(item["instruction"])},
            {"role": "user", "content": to_str(item["input"])},
            {"role": "assistant", "content": to_str(item["output"])},
        ]
        return {
            "full_text": tok.apply_chat_template(messages, tokenize=False),
            "prompt_text": tok.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True),
        }

    ds = Dataset.from_list([make_example(it) for it in data])

    def tokenize(ex):
        full_ids = tok(ex["full_text"], truncation=True, max_length=MAX_LEN,
                       add_special_tokens=False)["input_ids"]
        prompt_ids = tok(ex["prompt_text"], truncation=True, max_length=MAX_LEN,
                         add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        return {"input_ids": full_ids, "labels": labels}

    tok_ds = ds.map(tokenize, remove_columns=["full_text", "prompt_text"],
                    desc="Tokenizing")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, padding=True, label_pad_token_id=-100,
        return_tensors="pt")

    steps = max(1, len(data) // (BATCH_SIZE * GRAD_ACCUM))
    warmup = max(1, int(steps * EPOCHS * 0.03))

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, logging_steps=10, save_steps=200,
        save_total_limit=2, warmup_steps=warmup,
        lr_scheduler_type="cosine", bf16=True, report_to="none",
        dataloader_drop_last=False, remove_unused_columns=False)

    trainer = Trainer(model=model, args=args, train_dataset=tok_ds,
                      data_collator=collator)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    del model, trainer
    gc.collect(); torch.cuda.empty_cache()
    print(f"[DONE] {MODEL_KEY} | {group} | seed{seed}", flush=True)


if __name__ == "__main__":
    total = len(GROUPS) * len(SEEDS)
    print(f"Total tasks: {total}", flush=True)
    for i, (g, s) in enumerate([(g, s) for g in GROUPS for s in SEEDS], 1):
        try:
            train_one(g, s)
        except Exception as e:
            print(f"[FAIL] {MODEL_KEY} | {g} | seed{s}: {e}", flush=True)
        print(f"Progress: {i}/{total}", flush=True)
    print("\nAll done.", flush=True)
