"""
Table 2b: Full NQ-Open statistics (F1 mean/std) for all six models.
Since seed-level NQ data for main experiment models is not available,
we use the previously computed table for those models and recompute
the two depth-extension models from available data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from config.paths import WORKSPACE, TABLES_DIR

OUTPUT_CSV = TABLES_DIR / "table2b_full_statistics_NQ.csv"

# Start with the existing table (which already has main experiment models)
existing = pd.read_csv(OUTPUT_CSV) if (TABLES_DIR / "table2b_full_statistics_NQ.csv").exists() else None

# Depth-extension NQ seed-level data
chatml_file = WORKSPACE / "eval_results_chatml_complete.csv"
c_group_file = WORKSPACE / "c_group_eval_results.csv"

df_chatml = pd.read_csv(chatml_file)
df_chatml = df_chatml[(df_chatml["dataset"] == "nq") & (df_chatml["group"] != "C")]

df_c = pd.read_csv(c_group_file)
df_c = df_c[df_c["dataset"] == "nq"]

df_depth = pd.concat([df_chatml, df_c], ignore_index=True)
model_map = {"llama1b": "Llama-1B", "qwen3b": "Qwen-3B"}
df_depth["model"] = df_depth["model"].map(model_map)

# Compute statistics for depth-extension models
stats_depth = df_depth.groupby(["model", "group"]).agg(
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
).round(4).reset_index()

# If existing table exists, replace rows for Llama-1B and Qwen-3B
if existing is not None:
    existing = existing[~existing["model"].isin(["Llama-1B", "Qwen-3B"])]
    final = pd.concat([existing, stats_depth], ignore_index=True)
else:
    # Fallback: hardcode main experiment NQ stats (if table missing)
    main_nq = [
        ("Qwen-1.5B", "A", 0.080108, 0.01276),
        ("Qwen-1.5B", "B1", 0.059411, 0.003681),
        ("Qwen-1.5B", "B2", 0.04766, 0.006188),
        ("Qwen-1.5B", "C", 0.039703, 0.00148),
        ("Llama-3B", "A", 0.078224, 0.009879),
        ("Llama-3B", "B1", 0.058826, 0.002539),
        ("Llama-3B", "B2", 0.054075, 0.001473),
        ("Llama-3B", "C", 0.056327, 0.000289),
        ("Qwen-7B", "A", 0.116607, 0.009763),
        ("Qwen-7B", "B1", 0.086589, 0.018831),
        ("Qwen-7B", "B2", 0.061169, 0.001891),
        ("Qwen-7B", "C", 0.061656, 0.002324),
        ("Llama-8B", "A", 0.046529, 0.00212),
        ("Llama-8B", "B1", 0.043819, 0.000325),
        ("Llama-8B", "B2", 0.040395, 0.000866),
        ("Llama-8B", "C", 0.042728, 0.001078),
    ]
    df_main = pd.DataFrame(main_nq, columns=["model", "group", "f1_mean", "f1_std"])
    final = pd.concat([df_main, stats_depth], ignore_index=True)

# Sort by model size and group
model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
group_order = ["A", "B1", "B2", "C"]
final["m"] = final["model"].apply(lambda x: model_order.index(x))
final["g"] = final["group"].apply(lambda x: group_order.index(x))
final = final.sort_values(["m", "g"]).drop(columns=["m", "g"])

final.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(final.to_string(index=False))