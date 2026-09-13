"""
Convert MBPP jsonl files to Alpaca format json.
Input: train_code_{group}.jsonl (fields: task_id, code)
Output: train_code_{group}.json (fields: instruction, input, output)
"""

import os
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import (
    MBPP_PROCESSED_DIR, MBPP_ALPACA_DIR,
    MBPP_ALPACA_A, MBPP_ALPACA_B1,
    MBPP_ALPACA_B2_3B, MBPP_ALPACA_B2_8B, MBPP_ALPACA_B2_70B,
    MBPP_ALPACA_C,
)

# Mapping: input jsonl -> output json
CONVERSIONS = [
    ("train_code_A.jsonl", MBPP_ALPACA_A),
    ("train_code_B1.jsonl", MBPP_ALPACA_B1),
    ("train_code_B2_3B.jsonl", MBPP_ALPACA_B2_3B),
    ("train_code_B2_8B.jsonl", MBPP_ALPACA_B2_8B),
    ("train_code_B2_70B.jsonl", MBPP_ALPACA_B2_70B),
    ("train_code_C.jsonl", MBPP_ALPACA_C),
]

# Load clean file to get task text (instruction)
CLEAN_FILE = MBPP_PROCESSED_DIR / "train_code_clean.jsonl"
with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
    clean_data = [json.loads(line) for line in f if line.strip()]

task_id_to_text = {item['task_id']: item['text'] for item in clean_data}

INSTRUCTION_TEMPLATE = "Write a Python function for the following problem:\n{problem}"

for input_name, output_path in CONVERSIONS:
    input_path = MBPP_PROCESSED_DIR / input_name
    if not os.path.exists(input_path):
        print(f"[SKIP] {input_name} not found")
        continue

    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    alpaca_data = []
    for item in data:
        task_id = item['task_id']
        problem = task_id_to_text.get(task_id, "")
        alpaca_data.append({
            "instruction": INSTRUCTION_TEMPLATE.format(problem=problem),
            "input": "",
            "output": item['code'],
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(alpaca_data)} samples to {output_path}")

print("\nAll conversions completed.")