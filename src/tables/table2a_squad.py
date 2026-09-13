"""
Table 2a: Full SQuAD statistics (F1 mean/std, HER mean/std) for all six models.
Reads seed-level evaluation results from the workspace.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from config.paths import WORKSPACE, TABLES_DIR

OUTPUT_CSV = TABLES_DIR / "table2a_full_statistics_squad.csv"

# Main experiment seed-level files
main_files = {
    "Qwen-1.5B": WORKSPACE / "qwen_1.5b_2000_eval_results.csv",
    "Llama-3B": WORKSPACE / "llama32_2000_eval_results.csv",
    "Qwen-7B": WORKSPACE / "qwen2.5_7b_2000_eval_results.csv",
    "Llama-8B": WORKSPACE / "llama8b_eval_results.csv",
}

# Depth-extension seed-level files
chatml_file = WORKSPACE / "eval_results_chatml_complete.csv"
c_group_file = WORKSPACE / "c_group_eval_results.csv"

frames = []

# Process main experiment files
for model, path in main_files.items():
    df = pd.read_csv(path)
    df["model"] = model
    frames.append(df[["model", "group", "seed", "F1", "HER"]].rename(columns={"F1": "f1", "HER": "her"}))

# Process depth-extension: A/B1/B2 from chatml, C from c_group
df_chatml = pd.read_csv(chatml_file)
df_chatml = df_chatml[(df_chatml["dataset"] == "squad") & (df_chatml["group"] != "C")]

df_c = pd.read_csv(c_group_file)
df_c = df_c[df_c["dataset"] == "squad"]

df_depth = pd.concat([df_chatml, df_c], ignore_index=True)
model_map = {"llama1b": "Llama-1B", "qwen3b": "Qwen-3B"}
df_depth["model"] = df_depth["model"].map(model_map)
df_depth = df_depth[["model", "group", "seed", "f1", "her"]]
frames.append(df_depth)

df_all = pd.concat(frames, ignore_index=True)

# Group and compute statistics
stats = df_all.groupby(["model", "group"]).agg(
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    her_mean=("her", "mean"),
    her_std=("her", "std"),
).round(4).reset_index()

# Sort by model size order
model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
group_order = ["A", "B1", "B2", "C"]
stats["m"] = stats["model"].apply(lambda x: model_order.index(x))
stats["g"] = stats["group"].apply(lambda x: group_order.index(x))
stats = stats.sort_values(["m", "g"]).drop(columns=["m", "g"])

stats.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(stats.to_string(index=False))