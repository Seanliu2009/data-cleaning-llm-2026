"""
Generate Table 2b: NQ-Open full statistics (F1 only) for all models and cleaning strategies.

Columns:
- model: model name (lowercase, hyphen-separated)
- group: cleaning strategy (A, B1, B2, C)
- f1_mean: mean F1 score
- f1_std: standard deviation of F1
"""

import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import TABLES_DIR

# Raw data from NQ-Open experiments, ordered from smallest to largest model
data = [
    # Model, Group, F1 Mean, F1 Std
    ("Qwen-1.5B", "A", 0.08010779808778168, 0.012759827875308445),
    ("Qwen-1.5B", "B1", 0.059411028758749464, 0.003680592665311539),
    ("Qwen-1.5B", "B2", 0.04766029707908066, 0.006188084870919009),
    ("Qwen-1.5B", "C", 0.03970345539337366, 0.0014800636852078958),
    ("Llama-3.2-3B", "A", 0.07822374348348235, 0.009878735879704847),
    ("Llama-3.2-3B", "B1", 0.058826191434633426, 0.0025385801869245253),
    ("Llama-3.2-3B", "B2", 0.054075456356234994, 0.0014730502750365752),
    ("Llama-3.2-3B", "C", 0.056326697179697384, 0.0002891232668403038),
    ("Qwen-2.5-7B", "A", 0.11660700118398207, 0.009763019835575803),
    ("Qwen-2.5-7B", "B1", 0.0865886068279719, 0.018831349939744405),
    ("Qwen-2.5-7B", "B2", 0.06116906081340856, 0.0018908205685632573),
    ("Qwen-2.5-7B", "C", 0.0616564806193509, 0.0023241859424753502),
    ("Llama-8B", "A", 0.0465287257252046, 0.0021198350208824),
    ("Llama-8B", "B1", 0.0438185961108286, 0.0003252611565594),
    ("Llama-8B", "B2", 0.0403949096763666, 0.0008655442998521),
    ("Llama-8B", "C", 0.0427281627821527, 0.0010779671939885),
]

df = pd.DataFrame(data, columns=["model_raw", "group", "f1_mean", "f1_std"])

# Normalize model names: lowercase + hyphen-separated
df["model"] = df["model_raw"].str.lower().str.replace(" ", "-")

# Sort by model size (smallest to largest)
model_order = ["qwen-1.5b", "llama-3.2-3b", "qwen-2.5-7b", "llama-8b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["model", "group"]).reset_index(drop=True)

# Select columns
df = df[["model", "group", "f1_mean", "f1_std"]]

# Save to CSV
df.to_csv(TABLES_DIR / "table2b_full_statistics_NQ.csv", index=False)
print(f"[INFO] Table 2b saved to {TABLES_DIR / 'table2b_full_statistics_NQ.csv'}")