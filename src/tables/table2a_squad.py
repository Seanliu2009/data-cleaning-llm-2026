"""
Generate Table 2a: SQuAD 2.0 full statistics (F1 and HER) for all models and cleaning strategies.

Columns:
- model: model name (lowercase, hyphen-separated)
- group: cleaning strategy (A, B1, B2, C)
- f1_mean: mean F1 score
- f1_std: standard deviation of F1
- her_mean: mean hallucination error rate
- her_std: standard deviation of HER
"""

import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import TABLES_DIR

# Raw data from SQuAD 2.0 experiments, ordered from smallest to largest model
data = [
    # Model, Group, F1 Mean, F1 Std, HER Mean, HER Std
    ("Qwen-1.5B", "A", 0.4095, 0.0252, 0.1540, 0.0462),
    ("Qwen-1.5B", "B1", 0.3207, 0.0089, 0.1454, 0.0130),
    ("Qwen-1.5B", "B2", 0.3116, 0.0071, 0.1787, 0.0129),
    ("Qwen-1.5B", "C", 0.3652, 0.0114, 0.1444, 0.0156),
    ("Llama-3.2-3B", "A", 0.7143, 0.0104, 0.1992, 0.0119),
    ("Llama-3.2-3B", "B1", 0.6742, 0.0125, 0.1826, 0.0152),
    ("Llama-3.2-3B", "B2", 0.6308, 0.0270, 0.2666, 0.0344),
    ("Llama-3.2-3B", "C", 0.6581, 0.0187, 0.1653, 0.0151),
    ("Qwen-2.5-7B", "A", 0.7155, 0.0498, 0.1015, 0.0221),
    ("Qwen-2.5-7B", "B1", 0.5549, 0.0278, 0.0931, 0.0123),
    ("Qwen-2.5-7B", "B2", 0.5113, 0.0213, 0.0939, 0.0122),
    ("Qwen-2.5-7B", "C", 0.6440, 0.1060, 0.1072, 0.0217),
    ("Llama-8B", "A", 0.6549, 0.0146, 0.0981, 0.0056),
    ("Llama-8B", "B1", 0.6048, 0.0205, 0.1134, 0.0121),
    ("Llama-8B", "B2", 0.5767, 0.0176, 0.1520, 0.0142),
    ("Llama-8B", "C", 0.6559, 0.0173, 0.0985, 0.0077),
]

df = pd.DataFrame(data, columns=["model_raw", "group", "f1_mean", "f1_std", "her_mean", "her_std"])

# Normalize model names: lowercase + hyphen-separated
df["model"] = df["model_raw"].str.lower().str.replace(" ", "-")

# Sort by model size (smallest to largest)
model_order = ["qwen-1.5b", "llama-3.2-3b", "qwen-2.5-7b", "llama-8b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["model", "group"]).reset_index(drop=True)

# Select columns
df = df[["model", "group", "f1_mean", "f1_std", "her_mean", "her_std"]]

# Save to CSV
df.to_csv(TABLES_DIR / "table2a_full_statistics_squad.csv", index=False)
print(f"[INFO] Table 2a saved to {TABLES_DIR / 'table2a_full_statistics_squad.csv'}")