"""
Evaluate Qwen2.5-1.5B on NQ-Open.
Loads QLoRA adapters for each group and seed, computes F1 on dev set.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    QWEN_1_5B_PATH,
    QWEN_1_5B_ADAPTER,
    NQ_RESULTS,
    NQ_DEV_FILE,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = QWEN_1_5B_PATH
ADAPTER_BASE = QWEN_1_5B_ADAPTER
OUTPUT_DIR = NQ_RESULTS / "Qwen1.5B"
DEV_FILE = NQ_DEV_FILE

GROUPS = ["A", "B1", "B2", "C"]
SEEDS = [42, 43, 44, 45, 46]

BATCH_SIZE = 512
MAX_NEW_TOKENS = 30

# ------------------------------------------------------------------
# 4-bit quantization config
# ------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ------------------------------------------------------------------
# Load dev set
# ------------------------------------------------------------------
print("Loading dev set...")
with open(DEV_FILE, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)
print(f"Dev set size: {len(dev_data)}")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def compute_f1(pred, gt):
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens) if pred_tokens else 0
    r = len(common) / len(gt_tokens) if gt_tokens else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print("Evaluating Qwen1.5B on NQ-Open")
print(f"{'='*60}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load base model with 4-bit quantization
print("Loading base model with 4-bit quantization...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

results = []

for group in GROUPS:
    for seed in SEEDS:
        adapter_path = ADAPTER_BASE / group / f"seed_{seed}_adapter"
        if not adapter_path.exists():
            print(f"  [WARN] {group} seed {seed} not found, skipping")
            continue

        print(f"\n  Evaluating {group} seed {seed}...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()

        f1_scores = []

        for i in tqdm(range(0, len(dev_data), BATCH_SIZE), desc=f"  {group} seed {seed}", leave=False):
            batch = dev_data[i:i+BATCH_SIZE]
            prompts = [f"{item['instruction']}\n\n{item['input']}\n\nAnswer:" for item in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for idx, (item, pred) in enumerate(zip(batch, predictions)):
                if "Answer:" in pred:
                    pred_text = pred.split("Answer:")[-1].strip()
                else:
                    pred_text = pred.strip()
                best_f1 = compute_f1(pred_text, item['output'])
                f1_scores.append(best_f1)

        avg_f1 = np.mean(f1_scores)
        results.append({"model": "Qwen1.5B", "group": group, "seed": seed, "F1": avg_f1})
        print(f"    F1 = {avg_f1:.4f}")

        del model
        torch.cuda.empty_cache()

# Save results
df = pd.DataFrame(results)
csv_path = OUTPUT_DIR / "nq_eval_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

summary = df.groupby("group").agg({"F1": ["mean", "std"]}).round(4)
print(f"\nQwen1.5B Summary:")
print(summary)