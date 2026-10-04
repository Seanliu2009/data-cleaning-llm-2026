"""
NQ-Open full evaluation for qwen0.5b.

6 groups x 5 seeds = 30 adapters.
Output: EVAL_RESULTS_DIR / "qwen0.5b_nq_results.csv"

Usage:
  python eval_qwen0.5b_nq.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json, time, re, string, gc
import numpy as np, pandas as pd, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from paths import QWEN_0_5B_PATH, QWEN_0_5B_NQ_ADAPTER, NQ_DEV_FILE, EVAL_RESULTS_DIR

MODEL_KEY = "qwen0.5b"
MODEL_PATH = str(QWEN_0_5B_PATH)
ADAPTER_ROOT = str(QWEN_0_5B_NQ_ADAPTER)
EVAL_FILE = str(NQ_DEV_FILE)
OUT = str(EVAL_RESULTS_DIR / f"{MODEL_KEY}_nq_results.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
SEEDS = [42, 43, 44, 45, 46]

SYSTEM = ("Answer the question based on your knowledge. "
          "If you don't know, respond with 'I don't know.'")
MAX_NEW = 32
BATCH = 64



def to_str(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def normalize_answer(s):
    def rm_art(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def ws(t): return " ".join(t.split())
    def rm_p(t): return "".join(c for c in t if c not in set(string.punctuation))
    return ws(rm_art(rm_p(s.lower())))


def compute_f1(pred, gt):
    p = normalize_answer(pred).split()
    g = normalize_answer(gt).split()
    if not p or not g:
        return float(p == g)
    c = Counter(p) & Counter(g)
    n = sum(c.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def clean(pred):
    for sep in [".", "\n"]:
        if sep in pred:
            pred = pred.split(sep)[0]
    return pred.strip()


with open(EVAL_FILE) as f:
    samples = json.load(f)
samples = sorted(samples, key=lambda s: len(to_str(s["input"])))
print(f"Eval samples: {len(samples)}", flush=True)

done_df = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
done_set = set()
if not done_df.empty:
    for _, r in done_df.iterrows():
        done_set.add((r["group"], int(r["seed"])))
print(f"Already done: {len(done_set)}", flush=True)

results = done_df.to_dict("records") if not done_df.empty else []

todo = [(g, s) for g in GROUPS for s in SEEDS
        if (g, s) not in done_set
        and os.path.exists(os.path.join(ADAPTER_ROOT, g, f"seed_{s}",
                                        "adapter_model.safetensors"))]

if todo:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True)

    first_g, first_s = todo[0]
    first_adapter = os.path.join(ADAPTER_ROOT, first_g, f"seed_{first_s}")
    model = PeftModel.from_pretrained(base, first_adapter, adapter_name="current")
    model.eval()

    for idx, (group, seed) in enumerate(todo):
        adapter_path = os.path.join(ADAPTER_ROOT, group, f"seed_{seed}")
        print(f"  [EVAL] {MODEL_KEY} | {group} | seed{seed}", flush=True)
        t0 = time.time()

        if idx > 0:
            model.load_adapter(adapter_path, adapter_name="current", is_trainable=False)
        model.set_adapter("current")

        f1s = []
        for i in range(0, len(samples), BATCH):
            batch = samples[i:i+BATCH]
            prompts = []
            for s in batch:
                user_text = to_str(s["input"])
                msgs = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
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
                f1s.append(compute_f1(pred, to_str(s["output"])))

        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        el = time.time() - t0
        print(f"    F1={avg_f1:.4f} ({el:.1f}s)", flush=True)

        results.append({"model": MODEL_KEY, "task": "nq",
                        "group": group, "seed": seed, "F1": avg_f1})
        pd.DataFrame(results).to_csv(OUT, index=False)

        if idx < len(todo) - 1:
            model.delete_adapter("current")
        gc.collect(); torch.cuda.empty_cache()

    del model, base
    gc.collect(); torch.cuda.empty_cache()

print(f"\nAll done. Saved: {OUT}", flush=True)
