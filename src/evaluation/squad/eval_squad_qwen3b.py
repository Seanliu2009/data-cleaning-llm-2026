"""
SQuAD 2.0 evaluation for qwen3b.

Runs: 6 groups x 10 seeds = 60 evaluations.

Metrics:
  F1  : official SQuAD normalize, max over multiple answers
  HER : hallucination/over-answer rate on unanswerable questions

Format matches training exactly (verified at token level):
  system   : SQUAD_SYSTEM
  user     : "Context: {context}\nQuestion: {question}"
  tokenize with add_special_tokens=False

Adapter reuse: PeftModel wrapped once, adapters loaded/deleted by name.

CSV resume: re-run this script to continue from last saved row.

Usage:
  python eval_squad_qwen3b.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import time
import re
import string
import gc
import numpy as np
import pandas as pd
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from paths import (
    QWEN_3B_PATH, QWEN_3B_SQUAD_ADAPTER,
    SQUAD_EVAL_FILE, SQUAD_RESULTS, SQUAD_SYSTEM,
)

MODEL_KEY = "qwen3b"
MODEL_PATH = str(QWEN_3B_PATH)
ADAPTER_ROOT = str(QWEN_3B_SQUAD_ADAPTER)
EVAL_FILE = str(SQUAD_EVAL_FILE)
OUT_CSV = str(SQUAD_RESULTS / f"squad_full_results_{MODEL_KEY}_officialf1.csv")
GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
SEEDS = list(range(42, 52))
SYSTEM = SQUAD_SYSTEM
MAX_NEW = 64
BATCH = 64


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(pred, gt):
    p = normalize_answer(pred).split()
    g = normalize_answer(gt).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr = n / len(p)
    rc = n / len(g)
    return 2 * pr * rc / (pr + rc)


def is_refusal(t):
    pats = ["i don't know", "i do not know", "i dont know",
            "no answer", "not enough information",
            "cannot be determined", "unanswerable", "unknown"]
    return any(p in t.lower() for p in pats)


def clean(pred):
    for sep in [".", "\n"]:
        if sep in pred:
            pred = pred.split(sep)[0]
    return pred.strip()


def eval_one(model, tok, samples):
    f1s, hal, total_un = [], 0, 0
    for i in range(0, len(samples), BATCH):
        batch = samples[i:i+BATCH]
        prompts = []
        for s in batch:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"Context: {s['context']}\nQuestion: {s['question']}"},
            ]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=512,
                     add_special_tokens=False).to("cuda")
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=MAX_NEW,
                                  do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        input_len = inputs["input_ids"].shape[1]
        for j, s in enumerate(batch):
            pred = tok.decode(outs[j][input_len:], skip_special_tokens=True)
            pred = clean(pred)
            if s["is_impossible"]:
                total_un += 1
                if not is_refusal(pred):
                    hal += 1
            else:
                if s["answers"]:
                    f1s.append(max(compute_f1(pred, a) for a in s["answers"]))
    avg_f1 = float(np.mean(f1s)) if f1s else 0.0
    her = hal / total_un if total_un else 0.0
    return avg_f1, her


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    with open(EVAL_FILE) as f:
        samples = json.load(f)
    samples = sorted(samples, key=lambda s: len(s["context"]) + len(s["question"]))
    print(f"Eval samples: {len(samples)}", flush=True)

    done_df = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    done_set = set()
    if not done_df.empty:
        for _, r in done_df.iterrows():
            done_set.add((r["model"], r["group"], int(r["seed"])))
    print(f"Already done: {len(done_set)}", flush=True)

    results = done_df.to_dict("records") if not done_df.empty else []

    todo = [(g, s) for g in GROUPS for s in SEEDS
            if (MODEL_KEY, g, s) not in done_set
            and os.path.exists(os.path.join(ADAPTER_ROOT, g,
                                            f"seed_{s}",
                                            "adapter_model.safetensors"))]
    if not todo:
        print(f"[SKIP] {MODEL_KEY}: nothing to do", flush=True)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=True)

        first_group, first_seed = todo[0]
        first_adapter = os.path.join(ADAPTER_ROOT, first_group,
                                     f"seed_{first_seed}")
        model = PeftModel.from_pretrained(base, first_adapter,
                                          adapter_name="current")
        model.eval()

        for idx, (group, seed) in enumerate(todo):
            adapter_path = os.path.join(ADAPTER_ROOT, group, f"seed_{seed}")
            print(f"  [EVAL] {MODEL_KEY} | {group} | seed{seed}", flush=True)
            t0 = time.time()

            if idx > 0:
                model.load_adapter(adapter_path, adapter_name="current",
                                   is_trainable=False)
            model.set_adapter("current")

            avg_f1, her = eval_one(model, tok, samples)
            el = time.time() - t0
            print(f"    F1={avg_f1:.4f} HER={her:.4f} ({el:.1f}s)", flush=True)

            results.append({"model": MODEL_KEY, "task": "squad",
                            "group": group, "seed": seed,
                            "F1": avg_f1, "HER": her})
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

            if idx < len(todo) - 1:
                model.delete_adapter("current")
            gc.collect()
            torch.cuda.empty_cache()

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll done. Saved: {OUT_CSV}", flush=True)
