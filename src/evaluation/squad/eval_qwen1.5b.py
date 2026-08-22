"""
Evaluate Qwen2.5-1.5B on SQuAD 2.0.
Loads QLoRA adapters, computes F1 and HER on a fixed evaluation set.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    QWEN_1_5B_PATH,
    QWEN_1_5B_SQUAD_ADAPTER,
    SQUAD_EVAL_FILE,
    SQUAD_OUTPUT_DIR,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_MODEL_PATH = QWEN_1_5B_PATH
ADAPTER_BASE = QWEN_1_5B_SQUAD_ADAPTER
EVAL_FILE = SQUAD_EVAL_FILE
OUTPUT_DIR = SQUAD_OUTPUT_DIR / "Qwen1.5B"

GROUPS = ["A", "B1", "B2", "C"]
SEEDS = list(range(42, 52))
BATCH_SIZE = 128
MAX_NEW_TOKENS = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load fixed evaluation set
# ------------------------------------------------------------------
print("Loading fixed evaluation set...")
with open(EVAL_FILE, 'r') as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} samples.")
answerable = sum(1 for s in samples if not s["is_impossible"])
unanswerable = sum(1 for s in samples if s["is_impossible"])
print(f"Answerable: {answerable}, Unanswerable: {unanswerable}")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def trim_to_answer(text):
    for sep in ['.', '\n', '?', '!']:
        if sep in text:
            text = text.split(sep)[0]
    return text.strip()

def extract_assistant_response(full_output):
    if "assistant" in full_output:
        parts = full_output.rsplit("assistant", 1)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if candidate.startswith("\n"):
                candidate = candidate[1:].strip()
            if candidate.startswith(":"):
                candidate = candidate[1:].strip()
            return candidate
    lines = full_output.split("\n")
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return full_output.strip()

def compute_f1(pred, gt):
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens) if pred_tokens else 0
    r = len(common) / len(gt_tokens) if gt_tokens else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def is_refusal(text):
    patterns = ["i don't know", "我不知道", "no answer", "not enough information", "cannot be determined"]
    return any(p in text.lower() for p in patterns)

# ------------------------------------------------------------------
# Load base model
# ------------------------------------------------------------------
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# ------------------------------------------------------------------
# Evaluate all
# ------------------------------------------------------------------
results = []
for group in GROUPS:
    for seed in tqdm(SEEDS, desc=f"Evaluating {group}"):
        adapter_path = ADAPTER_BASE / group / f"seed_{seed}_adapter"
        if not adapter_path.exists():
            print(f"Adapter not found: {adapter_path}")
            continue
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()

        f1_scores = []
        hallucination_count = 0
        total_unanswerable = 0

        for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Seed {seed}", leave=False):
            batch = samples[i:i+BATCH_SIZE]
            prompts = []
            for s in batch:
                messages = [
                    {"role": "system", "content": "Answer the question based on the given context. If the answer is not in the context, respond with 'I don't know.'"},
                    {"role": "user", "content": f"Context: {s['context']}\nQuestion: {s['question']}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for idx, (sample, pred) in enumerate(zip(batch, predictions)):
                pred_text = extract_assistant_response(pred)
                pred_text = trim_to_answer(pred_text)

                if sample["is_impossible"]:
                    total_unanswerable += 1
                    if not is_refusal(pred_text):
                        hallucination_count += 1
                else:
                    if sample["answers"]:
                        best_f1 = max(compute_f1(pred_text, ans) for ans in sample["answers"])
                        f1_scores.append(best_f1)

        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        her = hallucination_count / total_unanswerable if total_unanswerable > 0 else 0.0
        results.append({"group": group, "seed": seed, "F1": avg_f1, "HER": her})
        print(f"{group} seed {seed}: F1={avg_f1:.4f}, HER={her:.4f}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
df = pd.DataFrame(results)
summary = df.groupby("group").agg({"F1": ["mean", "std"], "HER": ["mean", "std"]}).round(4)
print("\n" + "="*60)
print("Qwen 1.5B Evaluation Summary (2000 samples, 10 seeds)")
print("="*60)
print(summary)

df.to_csv(OUTPUT_DIR / "qwen_1.5b_2000_eval_results.csv", index=False)
print(f"\nResults saved to {OUTPUT_DIR / 'qwen_1.5b_2000_eval_results.csv'}")