"""
Read the manually cleaned Excel sheet and apply changes to the noisy dataset.
Assumes you have overwritten modified_context cells with corrected values.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "squad"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SHEET_PATH = PROCESSED_DIR / "manual_cleaning_sheet.xlsx"
INPUT_PATH = PROCESSED_DIR / "train_2000_noisy.json"
OUTPUT_PATH = PROCESSED_DIR / "train_2000_C_manual.json"

print("Loading manual cleaning sheet...")
df = pd.read_excel(SHEET_PATH)

print(f"Total rows in sheet: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

cleaning_map = {}
modified_count = 0

for _, row in df.iterrows():
    idx = row['index']
    original = row['original_context']
    modified = row['modified_context']

    if modified != original:
        cleaning_map[idx] = modified
        modified_count += 1

print(f"Samples modified by manual cleaning: {modified_count}")

print("Loading noisy dataset...")
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

for i, sample in enumerate(data):
    if i in cleaning_map:
        sample['context'] = cleaning_map[i]

print(f"Applied corrections to {len(cleaning_map)} samples")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to: {OUTPUT_PATH}")
print("Done.")