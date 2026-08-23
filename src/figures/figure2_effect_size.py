"""
Generate Figure 2: Cohen's d (HER: C vs A) as a function of model size.
Data source: table4_statistical_summary_squad.csv (Table 4)
Models are ordered from smallest to largest.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# ------------------------------------------------------------------
# 1. Add project root to path so config can be imported
# ------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import FIGURES_DIR, TABLES_DIR

# ------------------------------------------------------------------
# 2. Load data from Table 4
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table4_statistical_summary_squad.csv")

# Model order: smallest to largest
model_order = ["qwen-1.5b", "llama-3.2-3b", "qwen-2.5-7b", "llama-8b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model").reset_index(drop=True)

models = df["model"].values
cohen_d_her = df["c_vs_a_her_d"].values

# Display names for the plot
display_names = ["Qwen-1.5B", "Llama-3.2-3B", "Qwen-2.5-7B", "Llama-8B"]

# ------------------------------------------------------------------
# 3. Create figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
bars = ax.bar(display_names, cohen_d_her, color=colors, edgecolor='black', linewidth=0.8)

# Reference line at y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel("Cohen's d (HER: C vs A)", fontsize=12)
ax.set_title("Effect Size of Manual Cleaning on HER vs Model Size", fontsize=13, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(-4.0, 1.0)

# Add value labels on top/below bars
for i, v in enumerate(cohen_d_her):
    offset = 0.08 if v >= 0 else -0.08
    va = 'bottom' if v >= 0 else 'top'
    ax.text(i, v + offset, f'{v:.3f}', ha='center', va=va, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure2_effect_size.png", dpi=300)
print(f"[INFO] Figure 2 saved to {FIGURES_DIR / 'figure2_effect_size.png'}")
plt.show()