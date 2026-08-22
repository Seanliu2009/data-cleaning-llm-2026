"""
Use Llama-3.2-3B to perform semantic cleaning on NQ questions.
Input: JSONL with 'question' and 'answer' fields.
Output: JSONL with cleaned 'question' field.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import LLAMA_3_2_3B_PATH, DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "nq"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = LLAMA_3_2_3B_PATH
INPUT_PATH = PROCESSED_DIR / "train_nq_A.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "train_nq_B2.jsonl"
LOG_PATH = PROCESSED_DIR / "modification_log_nq_B2.json"

BATCH_SIZE = 32

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

# ------------------------------------------------------------------
# Load model (FP16, no quantization)
# ------------------------------------------------------------------
print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True,
)
model.eval()
print("Model loaded on GPU.")

# ------------------------------------------------------------------
# Load JSONL data
# ------------------------------------------------------------------
print(f"Loading data from: {INPUT_PATH}")
data = []
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"Total samples: {len(data)}")

# ------------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a text cleaning expert. Your task is to correct semantic errors in the given question.\n"
    "Fix the following types of errors:\n"
    "1. Negation: remove wrongly inserted 'not' or 'never', or add back missing ones.\n"
    "2. Antonym: replace wrongly used antonyms.\n"
    "3. Entity/number swap: correct wrong numbers or entity names.\n"
    "4. Mutual exclusion: remove absolute qualifiers like 'always', 'never', 'unconditionally' if they break logic.\n"
    "ONLY output the corrected question. Do not add any explanation."
)

def build_prompt(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Correct this question:\n\n{question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ------------------------------------------------------------------
# Process data
# ------------------------------------------------------------------
modified_log = []
modified_count = 0

print("Starting LLM cleaning with Llama-3.2-3B...")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i+BATCH_SIZE]
        questions = [item['question'] for item in batch]
        prompts = [build_prompt(q) for q in questions]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for idx, (item, gen) in enumerate(zip(batch, generated)):
            parts = gen.split("assistant")
            cleaned = parts[-1].strip() if len(parts) > 1 else gen.strip()

            if cleaned != questions[idx]:
                modified_count += 1
                modified_log.append({
                    'index': i + idx,
                    'original_question': questions[idx],
                    'cleaned_question': cleaned
                })

            output_item = {
                'question': cleaned,
                'answer': item.get('answer', [])
            }
            f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')

print(f"\nModified {modified_count} out of {len(data)} samples.")

with open(LOG_PATH, 'w', encoding='utf-8') as f:
    json.dump(modified_log, f, ensure_ascii=False, indent=2)

print(f"B2 data saved to: {OUTPUT_PATH}")
print(f"Modification log saved to: {LOG_PATH}")
print("Done.")