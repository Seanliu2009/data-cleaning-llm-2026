"""
Prepare MBPP clean data.
Reads the raw mbpp.jsonl and saves all samples as train_code_clean.jsonl.
No sampling is performed because MBPP only has 974 samples.
"""

import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import MBPP_RAW_DIR, MBPP_CLEAN_FILE

RAW_FILE = MBPP_RAW_DIR / "mbpp.jsonl"
OUTPUT_FILE = MBPP_CLEAN_FILE

print(f"Loading raw MBPP from: {RAW_FILE}")
with open(RAW_FILE, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

print(f"Total samples: {len(raw_data)}")

clean_data = []
for item in raw_data:
    clean_data.append({
        'task_id': item['task_id'],
        'text': item['text'],
        'code': item['code'],
        'test_list': item.get('test_list', []),
        'test_setup_code': item.get('test_setup_code', ''),
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in clean_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Saved {len(clean_data)} samples to: {OUTPUT_FILE}")