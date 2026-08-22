"""
Convert SQuAD JSON files to Alpaca format JSON.
Input: JSON files with SQuAD format (context, question, answers, is_impossible)
Output: Alpaca format JSON files (instruction, input, output)
"""

import json
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "squad"
OUTPUT_DIR = DATA_DIR / "alpaca" / "squad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

source_files = {
    "A": PROCESSED_DIR / "train_2000_A_noisy.json",
    "B1": PROCESSED_DIR / "train_2000_B1_auto.json",
    "B2": PROCESSED_DIR / "train_2000_B2_llama.json",
    "C": PROCESSED_DIR / "train_2000_C_manual.json",
}

instruction = "Answer the question based on the given context. If the answer is not in the context, respond with 'I don't know.'"

for group, input_path in source_files.items():
    if not input_path.exists():
        print(f"[WARN] {input_path} not found, skipping {group}")
        continue

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    converted = []
    for item in raw_data:
        if item.get('is_impossible', False):
            answer = "I don't know."
        else:
            if 'answers' in item and item['answers']:
                if isinstance(item['answers'][0], dict):
                    answer = item['answers'][0].get('text', "I don't know.")
                else:
                    answer = item['answers'][0]
            else:
                answer = "I don't know."

        converted.append({
            "instruction": instruction,
            "input": f"Context: {item['context']}\nQuestion: {item['question']}",
            "output": answer
        })

    output_path = OUTPUT_DIR / f"train_{group}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"{group}: {len(converted)} samples saved to {output_path}")

print("All Alpaca data files created successfully!")