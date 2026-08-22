"""
Evaluate Llama-8B on SQuAD 2.0.
Loads QLoRA adapters, computes F1 and HER on a fixed evaluation set.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    LLAMA_8B_PATH,
    LLAMA_8B_SQUAD_ADAPTER,
    SQUAD_EVAL_FILE,
    SQUAD_OUTPUT_DIR,
)

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def trim_to_answer(text):
    for sep in [".", "\n", "?", "!"]:
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
# Evaluation Function
# ------------------------------------------------------------------
def evaluate_seed(model_path, adapter_path, eval_file, group, seed, batch_size=4, max_new_tokens=30):
    print(f"Evaluating {group} seed {seed}...")

    with open(eval_file, 'r') as f:
        samples = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    f1_scores = []
    hallucination_count = 0
    total_unanswerable = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
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
                max_new_tokens=max_new_tokens,
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
    print(f"{group} seed {seed}: F1={avg_f1:.4f}, HER={her:.4f}")
    return {"group": group, "seed": seed, "F1": avg_f1, "HER": her}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama-8B adapters on SQuAD 2.0")
    parser.add_argument("--model_path", type=str, default=str(LLAMA_8B_PATH), help="Path to base model")
    parser.add_argument("--adapter_base", type=str, default=str(LLAMA_8B_SQUAD_ADAPTER),
                        help="Base path where adapters are stored")
    parser.add_argument("--eval_file", type=str, default=str(SQUAD_EVAL_FILE), help="Path to evaluation set")
    parser.add_argument("--output_dir", type=str, default=str(SQUAD_OUTPUT_DIR / "Llama-8B"),
                        help="Directory to save results")
    parser.add_argument("--group", type=str, default=None, help="Group to evaluate (A, B1, B2, C)")
    parser.add_argument("--seed", type=int, default=None, help="Seed to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all groups and seeds")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Max tokens to generate")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        groups = ["A", "B1", "B2", "C"]
        seeds = list(range(42, 52))
        results = []
        for group in groups:
            for seed in seeds:
                adapter_path = Path(args.adapter_base) / group / f"seed_{seed}_adapter"
                if not adapter_path.exists():
                    print(f"Adapter not found: {adapter_path}")
                    continue
                res = evaluate_seed(
                    args.model_path,
                    str(adapter_path),
                    args.eval_file,
                    group,
                    seed,
                    args.batch_size,
                    args.max_new_tokens
                )
                results.append(res)

        df = pd.DataFrame(results)
        summary = df.groupby("group").agg({"F1": ["mean", "std"], "HER": ["mean", "std"]}).round(4)
        print("\n" + "="*60)
        print("Evaluation Summary (All Groups, 10 Seeds)")
        print("="*60)
        print(summary)
        df.to_csv(output_dir / "results.csv", index=False)

        groups = summary.index.tolist()
        f1_means = summary["F1"]["mean"].values
        f1_stds = summary["F1"]["std"].values
        her_means = summary["HER"]["mean"].values
        her_stds = summary["HER"]["std"].values
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(groups, f1_means, yerr=f1_stds, capsize=5, color=colors)
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 by Cleaning Strategy (10 seeds)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "f1_comparison.png", dpi=300)
        print(f"F1 plot saved to {output_dir / 'f1_comparison.png'}")

        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(groups, her_means, yerr=her_stds, capsize=5, color=colors)
        ax.set_ylabel("Hallucination Rate (HER)")
        ax.set_title("HER by Cleaning Strategy (10 seeds)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "her_comparison.png", dpi=300)
        print(f"HER plot saved to {output_dir / 'her_comparison.png'}")

    elif args.group is not None and args.seed is not None:
        adapter_path = Path(args.adapter_base) / args.group / f"seed_{args.seed}_adapter"
        if not adapter_path.exists():
            print(f"Adapter not found: {adapter_path}")
            exit(1)
        res = evaluate_seed(
            args.model_path,
            str(adapter_path),
            args.eval_file,
            args.group,
            args.seed,
            args.batch_size,
            args.max_new_tokens
        )
        df = pd.DataFrame([res])
        df.to_csv(output_dir / f"result_{args.group}_seed{args.seed}.csv", index=False)
        print(f"Result saved to {output_dir / f'result_{args.group}_seed{args.seed}.csv'}")
    else:
        print("Usage: python eval_llama8b.py [--all | --group A --seed 42]")