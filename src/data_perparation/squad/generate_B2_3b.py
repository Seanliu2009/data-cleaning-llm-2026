"""
Use Llama-3.2-3B-Instruct to perform semantic cleaning on B1 data (SQuAD).
Input: train_2000_B1_auto.json (original SQuAD format)
Output: train_2000_B2_llama.json + modification log
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import LLAMA_3_2_3B_PATH, DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "squad"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = LLAMA_3_2_3B_PATH
INPUT_PATH = PROCESSED_DIR / "train_2000_B1_auto.json"
OUTPUT_PATH = PROCESSED_DIR / "train_2000_B2_llama.json"
LOG_PATH = PROCESSED_DIR / "llama_modification_log.json"

BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

# ------------------------------------------------------------------
# Load model
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
# Load data
# ------------------------------------------------------------------
print(f"Loading data from: {INPUT_PATH}")
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Total samples: {len(data)}")

# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a text cleaning expert. Your task is to correct semantic errors in the given text.\n"
    "Fix the following types of errors:\n"
    "1. Negation: remove wrongly inserted 'not' or 'never', or add back missing ones.\n"
    "2. Antonym: replace wrongly used antonyms (e.g., 'decline' should be 'spread').\n"
    "3. Entity/number swap: correct wrong numbers or entity names.\n"
    "4. Mutual exclusion: remove absolute qualifiers like 'always', 'never', 'unconditionally' if they break logic.\n"
    "ONLY output the corrected text. Do not add any explanation."
)

def build_prompt(context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Correct this text:\n\n{context}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ------------------------------------------------------------------
# Process
# ------------------------------------------------------------------
modified_log = []
output_data = []
modified_count = 0

print("Starting LLM cleaning with Llama...")
for i in tqdm(range(0, len(data), BATCH_SIZE)):
    batch = data[i:i+BATCH_SIZE]
    contexts = [item['context'] for item in batch]
    prompts = [build_prompt(ctx) for ctx in contexts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for idx, (item, gen) in enumerate(zip(batch, generated)):
        parts = gen.split("assistant\n")
        corrected = parts[-1].strip() if len(parts) > 1 else gen.strip()

        if corrected != contexts[idx]:
            modified_count += 1
            modified_log.append({
                'index': i + idx,
                'id': item.get('id', 'unknown'),
                'original_context': contexts[idx],
                'modified_context': corrected
            })
        item['context'] = corrected
        output_data.append(item)

print(f"\nModified {modified_count} out of {len(data)} samples.")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

with open(LOG_PATH, 'w', encoding='utf-8') as f:
    json.dump(modified_log, f, ensure_ascii=False, indent=2)

print(f"B2 data saved to: {OUTPUT_PATH}")
print(f"Modification log saved to: {LOG_PATH}")
print("Done.")