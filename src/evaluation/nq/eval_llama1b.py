"""
Evaluate Llama-3.2-1B on NQ-Open for all cleaning strategies.
Groups: A, B1, B2, C
Seeds: 42-46 (5 seeds)
Evaluation format: ChatML
Metric: F1
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc

from config.paths import (
    LLAMA_1B_PATH,
    LLAMA_1B_NQ_ADAPTER,
    NQ_DEV_FILE,
    NQ_RESULTS,
)

MODEL_KEY = "llama1b"
MODEL_PATH = str(LLAMA_1B_PATH)
ADAPTER_BASE = LLAMA_1B_NQ_ADAPTER
EVAL_FILE = str(NQ_DEV_FILE)
OUTPUT_CSV = NQ_RESULTS / "eval_llama1b_nq_results.csv"

GROUPS = ["A", "B1", "B2", "C"]
SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 16
MAX_NEW_TOKENS = 30

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

def evaluate_adapter(adapter_path):
    with open(EVAL_FILE) as f:
        samples = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if not os.path.exists(adapter_path):
        print(f"Adapter not found: {adapter_path}")
        return None

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    f1_scores = []

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i+BATCH_SIZE]
        prompts = []
        golds = []

        for s in batch:
            messages = [{"role": "user", "content": s['instruction']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            golds.append(s['output'])

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred, gold in zip(predictions, golds):
            pred_text = extract_assistant_response(pred)
            pred_text = trim_to_answer(pred_text)
            f1_scores.append(compute_f1(pred_text, gold))

    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"f1": avg_f1}

if __name__ == "__main__":
    results = []
    for group in GROUPS:
        for seed in SEEDS:
            adapter_path = ADAPTER_BASE / group / f"seed_{seed}"
            if not os.path.exists(adapter_path):
                print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed} adapter not found")
                continue
            print(f"[EVAL] {MODEL_KEY} | {group} | seed{seed}")
            result = evaluate_adapter(str(adapter_path))
            if result:
                result["model"] = MODEL_KEY
                result["group"] = group
                result["seed"] = seed
                results.append(result)
                print(f"  F1={result['f1']:.4f}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV}")
    else:
        print("\nNo results collected.")