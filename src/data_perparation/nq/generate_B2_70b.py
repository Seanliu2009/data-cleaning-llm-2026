"""
Clean NQ A-group data using Llama-70B.
Output: train_nq_B2_70B.json
Only works on AutoDL where LLAMA_70B_PATH is available.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from config.paths import (
    NQ_DATA_DIR,
    LLAMA_70B_PATH,
    NQ_B2_70B_PATH,
)

INPUT_PATH = NQ_DATA_DIR / "train_nq_A.json"
OUTPUT_PATH = NQ_B2_70B_PATH
MAX_NEW_TOKENS = 512
BATCH_SIZE = 2

SYSTEM_PROMPT = (
    "You are a text cleaning expert. Your task is to correct semantic errors in the given text.\n"
    "Fix the following types of errors:\n"
    "1. Negation: remove wrongly inserted 'not' or 'never', or add back missing ones.\n"
    "2. Antonym: replace wrongly used antonyms (e.g., 'decline' should be 'spread').\n"
    "3. Entity/number swap: correct wrong numbers or entity names.\n"
    "4. Mutual exclusion: remove absolute qualifiers like 'always', 'never', 'unconditionally' if they break logic.\n"
    "ONLY output the corrected text. Do not add any explanation."
)

def main():
    if LLAMA_70B_PATH is None:
        print("70B model not available in this environment.")
        return
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}")
        return

    print(f"Loading model: {LLAMA_70B_PATH}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_70B_PATH), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        str(LLAMA_70B_PATH),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    def clean_batch(texts):
        prompts = []
        for text in texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Correct the following text:\n\n{text}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        cleaned = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            if "assistant" in decoded:
                decoded = decoded.split("assistant")[-1].strip()
                if decoded.startswith("\n"):
                    decoded = decoded[1:].strip()
                if decoded.startswith(":"):
                    decoded = decoded[1:].strip()
            cleaned.append(decoded)
        return cleaned

    cleaned_data = []
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Cleaning NQ with 70B"):
        batch = data[i:i+BATCH_SIZE]
        texts = [item["input"] for item in batch]
        cleaned_texts = clean_batch(texts)
        for item, cleaned_input in zip(batch, cleaned_texts):
            cleaned_data.append({
                "instruction": item["instruction"],
                "input": cleaned_input,
                "output": item["output"],
            })

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    print(f"Saved: {OUTPUT_PATH}")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()