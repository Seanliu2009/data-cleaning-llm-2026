"""
Table 1a: Simplified SQuAD statistics (A vs C) for all six models.
Reads table2a_full_statistics_squad.csv and extracts A and C columns.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from config.paths import TABLES_DIR

INPUT_CSV = TABLES_DIR / "table2a_full_statistics_squad.csv"
OUTPUT_CSV = TABLES_DIR / "table1a_squad_main.csv"

df = pd.read_csv(INPUT_CSV)

model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]

rows = []
for m in model_order:
    a_row = df[(df["model"] == m) & (df["group"] == "A")].iloc[0]
    c_row = df[(df["model"] == m) & (df["group"] == "C")].iloc[0]
    rows.append({
        "model": m,
        "A_F1": round(a_row["f1_mean"], 4),
        "C_F1": round(c_row["f1_mean"], 4),
        "A_HER": round(a_row["her_mean"], 4),
        "C_HER": round(c_row["her_mean"], 4),
    })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(out.to_string(index=False))