"""HumanEval generation for Llama-3.2-3B adapters (all groups and seeds)."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, gzip, torch, gc
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from config.paths import LLAMA_3_2_3B_PATH, HUMANEVAL_PATH, LLAMA_3_2_3B_CODE_ADAPTER, WORKSPACE

MODEL_KEY = "llama3b"
MODEL_PATH = str(LLAMA_3_2_3B_PATH)
ADAPTER_BASE = str(LLAMA_3_2_3B_CODE_ADAPTER)
OUTPUT_DIR = str(WORKSPACE / "code_eval_samples_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
SEEDS = [42, 43, 44, 45, 46]

MAX_NEW_TOKENS = 512
MAX_INPUT_LENGTH = 1024
BATCH_SIZE = 4

with gzip.open(HUMANEVAL_PATH, "rt", encoding="utf-8") as f:
    problems = [json.loads(line) for line in f]
print(f"HumanEval: {len(problems)} problems", flush=True)


def build_prompt(p):
    """Match the training prompt format (no system prompt for code)."""
    return f"<|user|>\n{p['prompt']}\n\n<|assistant|>\n"


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


for group in GROUPS:
    for seed in SEEDS:
        adapter_path = f"{ADAPTER_BASE}/{group}/seed_{seed}"
        out_file = os.path.join(OUTPUT_DIR, f"{MODEL_KEY}_{group}_seed{seed}.jsonl")

        if os.path.exists(out_file):
            print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed}", flush=True)
            continue
        if not os.path.exists(adapter_path):
            print(f"[SKIP] {MODEL_KEY} | {group} | seed{seed}: adapter missing", flush=True)
            continue

        print(f"[GEN] {MODEL_KEY} | {group} | seed{seed}", flush=True)

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=True)
        base.eval()
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        samples = []
        for i in tqdm(range(0, len(problems), BATCH_SIZE),
                      desc=f"  {group}-{seed}", leave=False):
            batch = problems[i:i+BATCH_SIZE]
            prompts = [build_prompt(p) for p in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_INPUT_LENGTH,
                               add_special_tokens=False).to("cuda")

            with torch.no_grad():
                outs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)

            input_len = inputs["input_ids"].shape[1]
            for j, p in enumerate(batch):
                completion = tokenizer.decode(
                    outs[j][input_len:], skip_special_tokens=True)
                samples.append({
                    "task_id": p["task_id"],
                    "entry_point": p["entry_point"],
                    "prompt": prompts[j],
                    "completion": completion,
                })

        with open(out_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        del model, base
        gc.collect(); torch.cuda.empty_cache()
        print(f"[DONE] {MODEL_KEY} | {group} | seed{seed}", flush=True)

print("\nAll done.", flush=True)