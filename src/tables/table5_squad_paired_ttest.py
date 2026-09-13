"""
Table 5: Paired t-test on SQuAD 2.0 (A vs C) for all six models.
Uses seed-level data from all models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from config.paths import WORKSPACE, TABLES_DIR

OUTPUT_CSV = TABLES_DIR / "table5_squad_paired_ttest_results.csv"

# Main experiment SQuAD seed-level data
main_files = {
    "Qwen-1.5B": WORKSPACE / "qwen_1.5b_2000_eval_results.csv",
    "Llama-3B": WORKSPACE / "llama32_2000_eval_results.csv",
    "Qwen-7B": WORKSPACE / "qwen2.5_7b_2000_eval_results.csv",
    "Llama-8B": WORKSPACE / "llama8b_eval_results.csv",
}

frames = []
for model, path in main_files.items():
    df = pd.read_csv(path)
    df["model"] = model
    frames.append(df[["model", "group", "seed", "F1"]].rename(columns={"F1": "f1"}))

# Depth-extension SQuAD seed-level data
df_chatml = pd.read_csv(WORKSPACE / "eval_results_chatml_complete.csv")
df_chatml = df_chatml[(df_chatml["dataset"] == "squad") & (df_chatml["group"] != "C")]

df_c = pd.read_csv(WORKSPACE / "c_group_eval_results.csv")
df_c = df_c[df_c["dataset"] == "squad"]

df_depth = pd.concat([df_chatml, df_c], ignore_index=True)
model_map = {"llama1b": "Llama-1B", "qwen3b": "Qwen-3B"}
df_depth["model"] = df_depth["model"].map(model_map)
df_depth = df_depth[["model", "group", "seed", "f1"]]
frames.append(df_depth)

df_all = pd.concat(frames, ignore_index=True)

# Compute paired t-test for each model
model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
rows = []
for m in model_order:
    sub_a = df_all[(df_all["model"] == m) & (df_all["group"] == "A")].sort_values("seed")
    sub_c = df_all[(df_all["model"] == m) & (df_all["group"] == "C")].sort_values("seed")
    if len(sub_a) < 2 or len(sub_c) < 2:
        continue
    merged = pd.merge(sub_a[["seed", "f1"]], sub_c[["seed", "f1"]], on="seed", suffixes=("_a", "_c"))
    t_stat, p_value = stats.ttest_rel(merged["f1_a"], merged["f1_c"])
    rows.append({
        "model": m,
        "mean_a": round(merged["f1_a"].mean(), 6),
        "std_a": round(merged["f1_a"].std(), 6),
        "mean_c": round(merged["f1_c"].mean(), 6),
        "std_c": round(merged["f1_c"].std(), 6),
        "diff": round((merged["f1_a"] - merged["f1_c"]).mean(), 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round((merged["f1_a"] - merged["f1_c"]).mean() / (merged["f1_a"] - merged["f1_c"]).std(), 4)
                    if (merged["f1_a"] - merged["f1_c"]).std() > 0 else 0,
        "n_seeds": len(merged),
        "is_significant": p_value < 0.05,
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(df_out.to_string(index=False))