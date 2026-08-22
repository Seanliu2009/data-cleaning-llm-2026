"""
Extract manually cleaned questions from Excel sheet and export as JSONL.
"""

import pandas as pd
import json
from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "nq"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = PROCESSED_DIR / "manual_cleaning_sheet_nq.xlsx"
OUTPUT_JSONL = PROCESSED_DIR / "train_nq_C.jsonl"

# ------------------------------------------------------------------
# Read Excel
# ------------------------------------------------------------------
print(f"Reading Excel from: {EXCEL_PATH}")
df = pd.read_excel(EXCEL_PATH, sheet_name='Cleaning Sheet')

print("Columns in the Excel file:")
print(df.columns.tolist())

# Expected column names (adjust if different)
col_cleaned = 'noisy_question'   # Column C with manually corrected question
col_answer = 'answer'            # Column E with answer

if col_cleaned not in df.columns:
    raise KeyError(f"Column '{col_cleaned}' not found. Available: {df.columns.tolist()}")
if col_answer not in df.columns:
    raise KeyError(f"Column '{col_answer}' not found. Available: {df.columns.tolist()}")

print(f"Loaded {len(df)} rows.")

records = []
for idx, row in df.iterrows():
    question = str(row[col_cleaned]).strip() if pd.notna(row[col_cleaned]) else ""
    ans = row[col_answer]
    if isinstance(ans, list):
        ans = ans[0] if ans else "I don't know."
    elif pd.isna(ans):
        ans = "I don't know."
    else:
        ans = str(ans).strip()

    records.append({
        'question': question,
        'answer': ans
    })

with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f"Saved {len(records)} records to: {OUTPUT_JSONL}")
print("Done.")