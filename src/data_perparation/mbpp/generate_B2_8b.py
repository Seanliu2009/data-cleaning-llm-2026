"""
Use Llama-8B to clean MBPP code (B2 group).
Output: train_code_B2_8B.jsonl
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import LLAMA_8B_PATH, MBPP_A_FILE, MBPP_B2_8B_FILE

BATCH_SIZE = 4
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a code cleaning expert. Your task is to correct semantic errors in the given code.\n"
    "Fix the following types of errors:\n"
    "1. Operator errors: wrong operators like '+' instead of '-', or '==' instead of '!='.\n"
    "2. Wrong numbers: incorrect numeric values.\n"
    "3. Extra statements: remove wrongly inserted 'pass' or 'return None'.\n"
    "4. Variable name errors: fix wrongly renamed variables.\n"
    "ONLY output the corrected code. Do not add any explanation."
)

def main():
    print(f"Loading model: {LLAMA_8B_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_8B_PATH), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        str(LLAMA_8B_PATH),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    with open(MBPP_A_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data)} samples")

    def clean_batch(codes):
        prompts = []
        for code in codes:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Correct the following code:\n\n{code}"}
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
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Cleaning MBPP with 8B"):
        batch = data[i:i+BATCH_SIZE]
        codes = [item['code'] for item in batch]
        cleaned_codes = clean_batch(codes)
        for item, cleaned_code in zip(batch, cleaned_codes):
            cleaned_data.append({
                'task_id': item['task_id'],
                'code': cleaned_code,
            })

    with open(MBPP_B2_8B_FILE, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved: {MBPP_B2_8B_FILE}")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()