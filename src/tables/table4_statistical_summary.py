"""
Generate Table 4: Statistical significance summary for SQuAD 2.0 experiments.

Columns:
- model: model name (lowercase, hyphen-separated)
- anova_f1_p: ANOVA p-value for F1 across cleaning strategies
- anova_her_p: ANOVA p-value for HER across cleaning strategies
- c_vs_a_f1_p: paired t-test p-value for C vs A on F1
- c_vs_a_f1_d: Cohen's d for C vs A on F1
- c_vs_a_her_p: paired t-test p-value for C vs A on HER
- c_vs_a_her_d: Cohen's d for C vs A on HER
"""

import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import TABLES_DIR

# Raw data from SQuAD 2.0 statistical tests, ordered from smallest to largest model
data = [
    # Model, ANOVA F1 p, ANOVA HER p, C vs A F1 p, C vs A F1 d, C vs A HER p, C vs A HER d
    ("Qwen-1.5B", "<0.001", 0.0191, 0.002, -1.3545, 0.4487, -0.2505),
    ("Llama-3.2-3B", "<0.001", "<0.001", 0.0001, -2.2312, 0, -3.1054),
    ("Qwen-2.5-7B", "<0.001", 0.2494, 0.1548, -0.4911, 0.648, 0.1493),
    ("Llama-8B", "<0.001", "<0.001", 0.7949, 0.0847, 0.7957, 0.0843),
]

df = pd.DataFrame(
    data,
    columns=[
        "model_raw",
        "anova_f1_p",
        "anova_her_p",
        "c_vs_a_f1_p",
        "c_vs_a_f1_d",
        "c_vs_a_her_p",
        "c_vs_a_her_d",
    ],
)

# Normalize model names: lowercase + hyphen-separated
df["model"] = df["model_raw"].str.lower().str.replace(" ", "-")

# Sort by model size (smallest to largest)
model_order = ["qwen-1.5b", "llama-3.2-3b", "qwen-2.5-7b", "llama-8b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model").reset_index(drop=True)

# Reorder columns
df = df[
    [
        "model",
        "anova_f1_p",
        "anova_her_p",
        "c_vs_a_f1_p",
        "c_vs_a_f1_d",
        "c_vs_a_her_p",
        "c_vs_a_her_d",
    ]
]

# Save to CSV
df.to_csv(TABLES_DIR / "table4_statistical_summary_squad.csv", index=False)
print(f"[INFO] Table 4 saved to {TABLES_DIR / 'table4_statistical_summary_squad.csv'}")