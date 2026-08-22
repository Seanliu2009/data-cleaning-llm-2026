"""
Generate Figure 2: Cohen's d (HER: C vs A) as a function of model size.
Reads data from table4_statistical_summary.csv.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import FIGURES_DIR, TABLES_DIR

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table4_statistical_summary.csv")

# Model order
model_order = ["qwen-1.5b", "llama-3.2-3b", "qwen-2.5-7b", "llama-8b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model").reset_index(drop=True)

models = df["model"].values
cohen_d_her = df["c_vs_a_her_d"].values

# ------------------------------------------------------------------
# 2. Create figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
bars = ax.bar(models, cohen_d_her, color=colors, edgecolor='black', linewidth=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel("Cohen's d (HER: C vs A)")
ax.set_title("Effect Size of Manual Cleaning on HER vs Model Size")
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(-4.0, 1.0)

# Add value labels on top of the bars
for i, v in enumerate(cohen_d_her):
    offset = 0.08 if v >= 0 else -0.08
    va = 'bottom' if v >= 0 else 'top'
    ax.text(i, v + offset, f'{v:.3f}', ha='center', va=va,
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure2_effect_size.png", dpi=300)
print(f"[INFO] Figure 2 saved to {FIGURES_DIR / 'figure2_effect_size.png'}")
plt.show()