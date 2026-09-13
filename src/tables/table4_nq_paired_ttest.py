"""
Table 4: Paired t-test on NQ-Open (A vs C) for all six models.
Uses seed-level data for depth-extension models; uses previously computed
values for main experiment models (no seed-level data available).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from config.paths import WORKSPACE, TABLES_DIR

OUTPUT_CSV = TABLES_DIR / "table4_nq_paired_ttest_results.csv"

# Depth-extension NQ seed-level data
df_chatml = pd.read_csv(WORKSPACE / "eval_results_chatml_complete.csv")
df_chatml = df_chatml[(df_chatml["dataset"] == "nq") & (df_chatml["group"] != "C")]

df_c = pd.read_csv(WORKSPACE / "c_group_eval_results.csv")
df_c = df_c[df_c["dataset"] == "nq"]

df_depth = pd.concat([df_chatml, df_c], ignore_index=True)
model_map = {"llama1b": "Llama-1B", "qwen3b": "Qwen-3B"}
df_depth["model"] = df_depth["model"].map(model_map)

# Compute t-test for depth-extension models
depth_rows = []
for m in ["Llama-1B", "Qwen-3B"]:
    sub_a = df_depth[(df_depth["model"] == m) & (df_depth["group"] == "A")].sort_values("seed")
    sub_c = df_depth[(df_depth["model"] == m) & (df_depth["group"] == "C")].sort_values("seed")
    if len(sub_a) < 2 or len(sub_c) < 2:
        continue
    merged = pd.merge(sub_a[["seed", "f1"]], sub_c[["seed", "f1"]], on="seed", suffixes=("_a", "_c"))
    t_stat, p_value = stats.ttest_rel(merged["f1_a"], merged["f1_c"])
    depth_rows.append({
        "model": m,
        "mean_a": round(merged["f1_a"].mean(), 6),
        "std_a": round(merged["f1_a"].std(), 6),
        "mean_c": round(merged["f1_c"].mean(), 6),
        "std_c": round(merged["f1_c"].std(), 6),
        "diff": round((merged["f1_a"] - merged["f1_c"]).mean(), 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "n_seeds": len(merged),
        "is_significant": p_value < 0.05,
    })

# Hardcoded main experiment results (from previous table)
main_rows = [
    {"model": "Qwen-1.5B", "mean_a": 0.080108, "std_a": 0.011413, "mean_c": 0.039703, "std_c": 0.001324,
     "diff": 0.040404, "t_stat": 7.3532, "p_value": 0.000911, "n_seeds": 5, "is_significant": True},
    {"model": "Llama-3B", "mean_a": 0.078224, "std_a": 0.008836, "mean_c": 0.056327, "std_c": 0.000259,
     "diff": 0.021897, "t_stat": 4.9032, "p_value": 0.004013, "n_seeds": 5, "is_significant": True},
    {"model": "Qwen-7B", "mean_a": 0.116607, "std_a": 0.008732, "mean_c": 0.061656, "std_c": 0.002079,
     "diff": 0.054951, "t_stat": 14.5310, "p_value": 0.000065, "n_seeds": 5, "is_significant": True},
    {"model": "Llama-8B", "mean_a": 0.046529, "std_a": 0.002120, "mean_c": 0.042728, "std_c": 0.001078,
     "diff": 0.003801, "t_stat": 3.0834, "p_value": 0.018405, "n_seeds": 5, "is_significant": True},
]

all_rows = depth_rows + main_rows
df_out = pd.DataFrame(all_rows)

model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
df_out["m"] = df_out["model"].apply(lambda x: model_order.index(x))
df_out = df_out.sort_values("m").drop(columns=["m"])

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(df_out.to_string(index=False))