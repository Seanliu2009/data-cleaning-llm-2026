"""
Generate an Excel spreadsheet with original and noisy code for manual cleaning.
"""

import json
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import MBPP_CLEAN_FILE, MBPP_A_FILE, MBPP_CLEANING_SHEET

with open(MBPP_CLEAN_FILE, 'r', encoding='utf-8') as f:
    clean_data = [json.loads(line) for line in f if line.strip()]

with open(MBPP_A_FILE, 'r', encoding='utf-8') as f:
    noisy_data = [json.loads(line) for line in f if line.strip()]

print(f"Clean samples: {len(clean_data)}, Noisy samples: {len(noisy_data)}")

rows = []
for i in range(len(clean_data)):
    rows.append({
        'index': i,
        'task_id': clean_data[i]['task_id'],
        'original_code': clean_data[i]['code'],
        'noisy_code': noisy_data[i]['code'],
        'cleaned_code': '',
    })

df = pd.DataFrame(rows)

try:
    from openpyxl import Workbook
    with pd.ExcelWriter(MBPP_CLEANING_SHEET, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Cleaning Sheet', index=False)
        worksheet = writer.sheets['Cleaning Sheet']
        for col in worksheet.columns:
            max_length = 0
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 100)
            worksheet.column_dimensions[col[0].column_letter].width = adjusted_width
except ImportError:
    print("openpyxl not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'openpyxl'])
    print("Please run the script again.")
    exit()

print(f"Saved to: {MBPP_CLEANING_SHEET}")
print(f"Total rows: {len(df)}")
print("Instructions:")
print("1. Open the Excel file.")
print("2. Compare original_code vs noisy_code.")
print("3. If you see a difference, type the corrected code in the 'cleaned_code' column.")
print("4. If no difference, leave blank (script will use original).")
print("5. Save the Excel file.")
print("6. Then run apply_manual_cleaning.py to generate C group data.")