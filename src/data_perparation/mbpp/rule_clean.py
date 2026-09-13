"""
Rule-based cleaning (B1) for MBPP code.
Fix indentation, remove trailing whitespace, normalize spaces.
Output: train_code_B1.jsonl
"""

import json
import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import MBPP_A_FILE, MBPP_B1_FILE

def clean_code(code):
    original = code
    # Normalize tabs to 4 spaces
    code = code.replace('\t', '    ')
    # Remove trailing whitespace per line
    code = '\n'.join([line.rstrip() for line in code.split('\n')])
    # Collapse multiple blank lines
    code = re.sub(r'\n\s*\n+', '\n\n', code)
    # Strip leading/trailing whitespace
    code = code.strip()
    changes = (code != original)
    return code, changes

with open(MBPP_A_FILE, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Total samples: {len(data)}")

cleaned_data = []
modified_count = 0
for item in data:
    new_item = item.copy()
    cleaned, changes = clean_code(item['code'])
    if changes:
        modified_count += 1
    new_item['code'] = cleaned
    cleaned_data.append(new_item)

print(f"Samples modified: {modified_count}")

with open(MBPP_B1_FILE, 'w', encoding='utf-8') as f:
    for item in cleaned_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Saved to: {MBPP_B1_FILE}")