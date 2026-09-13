"""
Extract manually cleaned code from Excel sheet and export as jsonl (C group).
"""

import pandas as pd
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import MBPP_CLEANING_SHEET, MBPP_C_FILE

print(f"Reading Excel from: {MBPP_CLEANING_SHEET}")
df = pd.read_excel(MBPP_CLEANING_SHEET, sheet_name='Cleaning Sheet')

print("Columns:", df.columns.tolist())

records = []
for _, row in df.iterrows():
    cleaned = str(row['cleaned_code']).strip() if pd.notna(row['cleaned_code']) else ""
    original = str(row['original_code']).strip()
    code = cleaned if cleaned else original
    records.append({
        'task_id': int(row['task_id']) if pd.notna(row['task_id']) else 0,
        'code': code,
    })

with open(MBPP_C_FILE, 'w', encoding='utf-8') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f"Saved {len(records)} records to: {MBPP_C_FILE}")