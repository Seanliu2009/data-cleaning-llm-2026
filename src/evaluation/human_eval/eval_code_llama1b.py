"""
Code generation evaluation for llama1b.

Output: EVAL_RESULTS_DIR / "code_eval_all/llama1b_{group}_seed{seed}.jsonl"

All models write to the same directory, prefixed with the model key.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gzip, json, time, gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from paths import LLAMA_1B_PATH, LLAMA_1B_CODE_ADAPTER, HUMANEVAL_PATH, EVAL_RESULTS_DIR

MODEL_KEY = "llama1b"
MODEL_PATH = str(LLAMA_1B_PATH)
ADAPTER_ROOT = str(LLAMA_1B_CODE_ADAPTER)
HUMANEVAL = str(HUMANEVAL_PATH)
OUT_DIR = str(EVAL_RESULTS_DIR / "code_eval_all")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
SEEDS = [42, 43, 44, 45, 46]

MAX_NEW = 512
BATCH = 8

humaneval = []
with gzip.open(HUMANEVAL, "rt") as f:
    for line in f:
        if line.strip():
            humaneval.append(json.loads(line))
print(f"HumanEval tasks: {len(humaneval)}", flush=True)

for f in os.listdir(OUT_DIR):
    if f.endswith(".jsonl.tmp"):
        os.remove(os.path.join(OUT_DIR, f))
        print(f"[CLEANED] {f}")

todo = [(g, s) for g in GROUPS for s in SEEDS
        if os.path.exists(os.path.join(ADAPTER_ROOT, g, f"seed_{s}",
                                        "adapter_model.safetensors"))
        and not os.path.exists(os.path.join(OUT_DIR, f"{MODEL_KEY}_{g}_seed{s}.jsonl"))]
print(f"Pending: {len(todo)}", flush=True)

if todo:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
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
        out_path = os.path.join(OUT_DIR, f"{MODEL_KEY}_{group}_seed{seed}.jsonl")
        tmp_path = out_path + ".tmp"
        print(f"  [EVAL] {MODEL_KEY} | {group} | seed{seed}", flush=True)
        t0 = time.time()

        if idx > 0:
            model.load_adapter(adapter_path, adapter_name="current", is_trainable=False)
        model.set_adapter("current")

        prompts_all, task_ids = [], []
        for item in humaneval:
            problem = item["prompt"]
            msgs = [{"role": "system", "content": problem},
                    {"role": "user", "content": ""}]
            prompts_all.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            task_ids.append(item["task_id"])

        completions = []
        for i in range(0, len(prompts_all), BATCH):
            batch = prompts_all[i:i+BATCH]
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=1024,
                         add_special_tokens=False).to("cuda")
            with torch.no_grad():
                outs = model.generate(**inputs, max_new_tokens=MAX_NEW,
                                      do_sample=False,
                                      pad_token_id=tok.eos_token_id)
            input_len = inputs["input_ids"].shape[1]
            for j in range(len(batch)):
                gen = tok.decode(outs[j][input_len:], skip_special_tokens=True)
                completions.append(gen)

        with open(tmp_path, "w") as f:
            for tid, comp in zip(task_ids, completions):
                f.write(json.dumps({"task_id": tid, "completion": comp}) + "\n")
        os.replace(tmp_path, out_path)

        el = time.time() - t0
        print(f"    Saved {out_path} ({el:.1f}s)", flush=True)

        if idx < len(todo) - 1:
            model.delete_adapter("current")
        gc.collect(); torch.cuda.empty_cache()

    del model, base
    gc.collect(); torch.cuda.empty_cache()

print(f"\nAll done. Output: {OUT_DIR}", flush=True)
