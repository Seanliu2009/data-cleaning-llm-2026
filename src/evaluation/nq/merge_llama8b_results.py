"""
Aggregate per-seed CSV results for Llama-8B on NQ-Open.
Reads all Llama-8B_*_seed*.csv files, computes per-group mean F1 and std,
and saves a summary CSV.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import NQ_RESULTS, OUTPUT_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RESULT_DIR = NQ_RESULTS / "Llama-8B"
OUTPUT_FILE = RESULT_DIR / "Llama-8B_NQ_summary.csv"

if not RESULT_DIR.exists():
    raise FileNotFoundError(f"Result directory not found: {RESULT_DIR}")

rows = []

for fname in sorted(os.listdir(RESULT_DIR)):
    if not fname.startswith("Llama-8B_") or not fname.endswith(".csv"):
        continue
    if fname == "Llama-8B_NQ_summary.csv":
        continue

    # Parse filename: Llama-8B_A_seed42.csv
    parts = fname.replace("Llama-8B_", "").replace(".csv", "").split("_seed")
    if len(parts) != 2:
        continue
    group = parts[0]
    seed = int(parts[1])

    df = pd.read_csv(RESULT_DIR / fname)
    f1_values = df['f1'].dropna()

    if len(f1_values) == 0:
        continue

    rows.append({
        "group": group,
        "seed": seed,
        "mean_f1": f1_values.mean(),
        "std_f1": f1_values.std(),
        "n": len(f1_values)
    })

if not rows:
    raise RuntimeError("No valid CSV files found.")

summary_df = pd.DataFrame(rows)
group_stats = summary_df.groupby("group").agg(
    mean_f1=("mean_f1", "mean"),
    std_f1=("mean_f1", "std"),
    n_seeds=("seed", "count")
).reset_index()

# Sort groups in logical order
group_order = {"A": 0, "B1": 1, "B2": 2, "C": 3}
group_stats["_order"] = group_stats["group"].map(group_order)
group_stats = group_stats.sort_values("_order").drop(columns=["_order"])

print("\n=== Llama-8B NQ-Open Summary ===")
print(group_stats.to_string(index=False))

group_stats.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved summary to: {OUTPUT_FILE}")