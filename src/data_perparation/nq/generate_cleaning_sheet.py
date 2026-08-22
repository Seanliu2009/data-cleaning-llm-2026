"""
Generate an Excel spreadsheet with original and noisy questions for manual cleaning.
"""

import json
import pandas as pd
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

CLEAN_FILE = PROCESSED_DIR / "train_nq_2000.jsonl"
NOISY_FILE = PROCESSED_DIR / "train_nq_A.jsonl"
OUTPUT_EXCEL = PROCESSED_DIR / "manual_cleaning_sheet_nq.xlsx"

print("Loading clean data...")
with open(CLEAN_FILE, 'r', encoding='utf-8') as f:
    clean_data = [json.loads(line) for line in f if line.strip()]

print("Loading noisy data...")
with open(NOISY_FILE, 'r', encoding='utf-8') as f:
    noisy_data = [json.loads(line) for line in f if line.strip()]

print(f"Clean samples: {len(clean_data)}, Noisy samples: {len(noisy_data)}")

rows = []
for i in range(len(clean_data)):
    rows.append({
        'index': i,
        'original_question': clean_data[i]['question'],
        'noisy_question': noisy_data[i]['question'],
        'cleaned_question': '',
        'answer': clean_data[i]['answer'][0] if clean_data[i].get('answer') and len(clean_data[i]['answer']) > 0 else "I don't know."
    })

df = pd.DataFrame(rows)

# Save to Excel
try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
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
            adjusted_width = min(max_length + 2, 80)
            worksheet.column_dimensions[col[0].column_letter].width = adjusted_width

except ImportError:
    print("openpyxl not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'openpyxl'])
    print("Please run the script again.")
    exit()

print(f"\nSaved to: {OUTPUT_EXCEL}")
print(f"Total rows: {len(df)}")
print("\nInstructions:")
print("1. Open the Excel file.")
print("2. Compare original_question vs noisy_question.")
print("3. If you see a difference, type the corrected version in the 'cleaned_question' column.")
print("4. If no difference, leave blank (script will use original).")
print("5. Save the Excel file.")
print("6. Then run apply_manual_cleaning.py to generate C group data.")