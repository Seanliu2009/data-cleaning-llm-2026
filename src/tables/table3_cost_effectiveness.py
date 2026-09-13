"""
Table 3: Cost effectiveness of cleaning strategies on SQuAD.
Computes small-model and large-model averages from table2a.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from config.paths import TABLES_DIR

INPUT_CSV = TABLES_DIR / "table2a_full_statistics_squad.csv"
OUTPUT_CSV = TABLES_DIR / "table3_cost_effectiveness_squad.csv"

df = pd.read_csv(INPUT_CSV)

small_models = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B"]
large_models = ["Qwen-7B", "Llama-8B"]

def avg(models, group, metric):
    return df[(df["model"].isin(models)) & (df["group"] == group)][metric].mean()

rows = []
for g in ["B1", "B2", "C"]:
    rows.append({
        "strategy": g,
        "small_model_delta_her": round(avg(small_models, "A", "her_mean") - avg(small_models, g, "her_mean"), 4),
        "small_model_delta_f1": round(avg(small_models, g, "f1_mean") - avg(small_models, "A", "f1_mean"), 4),
        "large_model_delta_her": round(avg(large_models, "A", "her_mean") - avg(large_models, g, "her_mean"), 4),
        "large_model_delta_f1": round(avg(large_models, g, "f1_mean") - avg(large_models, "A", "f1_mean"), 4),
    })

time_cost = {"B1": 0.45, "B2": 4.95, "C": 4.67}
recommendation = {
    "B1": "Small models only",
    "B2": "Not recommended",
    "C": "Small models only (fewer hallucinations but low F1)",
}

out = pd.DataFrame(rows)
out["est.time_h_"] = out["strategy"].map(time_cost)
out["recommendation"] = out["strategy"].map(recommendation)
out = out[["strategy", "est.time_h_", "small_model_delta_her", "small_model_delta_f1",
           "large_model_delta_her", "large_model_delta_f1", "recommendation"]]

out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(out.to_string(index=False))