"""
B2 prompt ablation: 3B x mbpp.

Cleans mbpp A-group data with 4 prompt variants.
Target field: output

Atomic write: .tmp + os.replace.
"""
import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from paths import LLAMA_3_2_3B_PATH, ABLATION_MBPP_DIR

MODEL_TAG = "3B"
MODEL_PATH = str(LLAMA_3_2_3B_PATH)
INPUT_PATH = "/mnt/workspace/code_data/train_code_A.json"
OUTPUT_DIR = str(ABLATION_MBPP_DIR)
FILE_PREFIX = "train_code_B2"
TARGET_FIELD = "output"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 512

CODE_PROMPTS = {
    "C1": (
        "You are a code cleaning expert. Your task is to correct semantic errors in the given code.\n"
        "Fix the following types of errors:\n"
        "1. Operator errors: wrong operators like '+' instead of '-', or '==' instead of '!='.\n"
        "2. Wrong numbers: incorrect numeric values.\n"
        "3. Extra statements: remove wrongly inserted 'pass' or 'return None'.\n"
        "4. Variable name errors: fix wrongly renamed variables.\n"
        "ONLY output the corrected code. Do not add any explanation."
    ),
    "C2": (
        "You are a code cleaning expert. Clean the given code by removing any errors you find. "
        "Preserve the original logic and structure.\n"
        "ONLY output the corrected code. Do not add any explanation."
    ),
    "C3": (
        "You are a code cleaning expert. Fix ONLY obvious syntax-level errors. "
        "If you are not confident, keep the original code unchanged. "
        "Do not refactor, do not add comments, do not wrap in markdown.\n"
        "ONLY output the corrected code. Do not add any explanation."
    ),
    "C4": (
        "You are a code cleaning expert. Clean the given code by fixing errors IN PLACE. "
        "Preserve the structure exactly. Do NOT wrap the output in markdown code fences. "
        "Do NOT add comments or docstrings.\n"
        "ONLY output the corrected code. Do not add any explanation."
    ),
}


def clean_dataset(model, tokenizer, data, prompt_text):
    n = len(data)
    out = []
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = data[i:i+BATCH_SIZE]
        texts = [item.get(TARGET_FIELD, "") for item in batch]
        prompts = []
        for t in texts:
            msgs = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": f"Correct the following:\n\n{t}"},
            ]
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        input_len = inputs["input_ids"].shape[1]
        for j, item in enumerate(batch):
            gen_ids = outputs[j][input_len:]
            cleaned = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            new_item = dict(item)
            new_item[TARGET_FIELD] = cleaned
            out.append(new_item)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean up stale .tmp files
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".json.tmp"):
            os.remove(os.path.join(OUTPUT_DIR, f))
            print(f"[CLEANED] {f}")

    print(f"Loading {MODEL_TAG}: {MODEL_PATH}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True)
    model.eval()

    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {INPUT_PATH}")

    for ptag, ptext in PROMPTS.items():
        out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}{ptag}_{MODEL_TAG}.json")
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path}")
            continue
        print(f"[RUN] {ptag}")
        cleaned = clean_dataset(model, tokenizer, data, ptext)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
        print(f"[SAVED] {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
