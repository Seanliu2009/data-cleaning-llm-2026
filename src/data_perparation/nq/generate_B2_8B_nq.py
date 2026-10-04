"""
B2 prompt ablation: 8B x nq.

Cleans nq A-group data with 4 prompt variants.
Target field: input

Atomic write: .tmp + os.replace.
"""
import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from paths import LLAMA_8B_PATH, ABLATION_NQ_DIR

MODEL_TAG = "8B"
MODEL_PATH = str(LLAMA_8B_PATH)
INPUT_PATH = "/mnt/workspace/nq_data/train_nq_A.json"
OUTPUT_DIR = str(ABLATION_NQ_DIR)
FILE_PREFIX = "train_nq_B2"
TARGET_FIELD = "input"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 512

TEXT_PROMPTS = {
    "C1": (
        "You are a text cleaning expert. Your task is to correct semantic errors in the given text.\n"
        "Fix the following types of errors:\n"
        "1. Negation: remove wrongly inserted 'not' or 'never', or add back missing ones.\n"
        "2. Antonym: replace wrongly used antonyms (e.g., 'decline' should be 'spread').\n"
        "3. Entity/number swap: correct wrong numbers or entity names.\n"
        "4. Mutual exclusion: remove absolute qualifiers like 'always', 'never', 'unconditionally' if they break logic.\n"
        "ONLY output the corrected text. Do not add any explanation."
    ),
    "C2": (
        "You are a text cleaning expert. Clean the given text by removing any errors you find. "
        "Preserve the original meaning and structure.\n"
        "ONLY output the corrected text. Do not add any explanation."
    ),
    "C3": (
        "You are a text cleaning expert. Fix ONLY obvious surface-level errors "
        "(spacing, punctuation, encoding). If you are not confident, keep the original text unchanged. "
        "Do not rephrase, do not change word order.\n"
        "ONLY output the corrected text. Do not add any explanation."
    ),
    "C4": (
        "You are a text cleaning expert. Clean the given text by fixing errors IN PLACE. "
        "Preserve the structure exactly: do not add, remove, or reorder any fields. "
        "Do not rewrite sentences that are already correct.\n"
        "ONLY output the corrected text. Do not add any explanation."
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
