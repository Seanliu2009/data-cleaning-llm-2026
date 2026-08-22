"""
Evaluate Llama-8B on NQ-Open.
Loads QLoRA adapters for each group and seed, computes F1 on dev set.
Uses command-line arguments for flexibility; paths default to config values.
"""

import json
import csv
import os
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import LLAMA_8B_PATH, LLAMA_8B_ADAPTER, NQ_RESULTS, NQ_DEV_FILE

# ------------------------------------------------------------------
# Normalization & F1
# ------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^a-zA-Z0-9]', ' ', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

def load_nq_data(data_path):
    with open(data_path, 'r') as f:
        raw = json.load(f)
    return raw

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama-8B on NQ-Open.")
    parser.add_argument('--base_model', type=str, default=str(LLAMA_8B_PATH),
                        help='Path to base model')
    parser.add_argument('--adapter_base', type=str, default=str(LLAMA_8B_ADAPTER),
                        help='Parent directory containing group subfolders (A, B1, B2, C)')
    parser.add_argument('--data_path', type=str, default=str(NQ_DEV_FILE),
                        help='Path to NQ dev JSON file')
    parser.add_argument('--output_dir', type=str, default=str(NQ_RESULTS / "Llama-8B"),
                        help='Where to save per-seed result CSVs')
    parser.add_argument('--groups', type=str, nargs='+', default=['A', 'B1', 'B2', 'C'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4-bit quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    raw_data = load_nq_data(args.data_path)
    questions = [item['input'] for item in raw_data]
    answers = [item['output'] for item in raw_data]
    print(f"Loaded {len(questions)} test samples.")

    for group in args.groups:
        for seed in args.seeds:
            adapter_dir = Path(args.adapter_base) / group / f"seed_{seed}_adapter"
            if not adapter_dir.exists():
                print(f"[WARN] Adapter not found: {adapter_dir}, skipping...")
                continue

            print(f"\n===== Evaluating group={group}, seed={seed} =====")
            model_peft = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
            model_peft.eval()

            f1_list = []
            pred_texts = []

            for i in tqdm(range(0, len(questions), args.batch_size), desc=f"{group}-s{seed}"):
                batch_q = questions[i:i+args.batch_size]
                batch_answers = answers[i:i+args.batch_size]

                inputs = tokenizer(
                    batch_q,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model_peft.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                for j, (q, gold) in enumerate(zip(batch_q, batch_answers)):
                    full_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
                    if full_text.startswith(q):
                        pred = full_text[len(q):].strip()
                    else:
                        pred = full_text.strip()
                    pred_texts.append(pred)
                    f1 = compute_f1(pred, gold)
                    f1_list.append(f1)

            csv_path = output_dir / f'Llama-8B_{group}_seed{seed}.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['question', 'prediction', 'gold_answer', 'f1'])
                for q, pred, gold, f1 in zip(questions, pred_texts, answers, f1_list):
                    writer.writerow([q, pred, gold, f1])

            avg_f1 = np.mean(f1_list)
            print(f"[{group}-s{seed}] Avg F1: {avg_f1:.4f}")

    print("\nAll evaluations finished.")

if __name__ == '__main__':
    main()