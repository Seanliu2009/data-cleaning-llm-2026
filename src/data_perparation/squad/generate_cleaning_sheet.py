"""
Generate a full Excel spreadsheet with original and modified context side by side for manual cleaning.
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

CLEAN_PATH = PROCESSED_DIR / "train_2000_clean.json"
NOISY_PATH = PROCESSED_DIR / "train_2000_noisy.json"
OUTPUT_PATH = PROCESSED_DIR / "manual_cleaning_sheet.xlsx"

print("Loading data...")

with open(CLEAN_PATH, 'r', encoding='utf-8') as f:
    clean_data = json.load(f)

with open(NOISY_PATH, 'r', encoding='utf-8') as f:
    noisy_data = json.load(f)

print(f"Total samples: {len(clean_data)}")

rows = []
for i in range(len(clean_data)):
    rows.append({
        'index': i,
        'id': clean_data[i]['id'],
        'original_context': clean_data[i]['context'],
        'modified_context': noisy_data[i]['context'],
    })

df = pd.DataFrame(rows)

print("\n--- Preview: first 2 samples ---")
for i in range(min(2, len(rows))):
    print(f"[{i}] original (first 100 chars): {rows[i]['original_context'][:100]}...")
    print(f"[{i}] modified (first 100 chars): {rows[i]['modified_context'][:100]}...")
    print("---")

# Save to Excel with formatting
try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Cleaning Sheet', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Cleaning Sheet']

        for col in worksheet.columns:
            max_length = 0
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[col[0].column_letter].width = adjusted_width

except ImportError:
    print("openpyxl not available, saving without formatting.")
    df.to_excel(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Total rows: {len(df)}")
print("\nHow to use:")
print("1. Open this Excel file.")
print("2. Compare original_context vs modified_context for each row.")
print("3. If you find a difference, copy original_context into modified_context.")
print("4. After all changes, save the file.")
print("5. Run apply_manual_cleaning.py to convert back to JSON.")