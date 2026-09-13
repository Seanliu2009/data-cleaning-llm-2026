"""
Figure 2: Effect size (Cohen's d) of manual cleaning on HER vs model size.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.paths import TABLES_DIR, FIGURES_DIR

TABLE_PATH = TABLES_DIR / "table2a_full_statistics_squad.csv"
OUTPUT_PATH = FIGURES_DIR / "figure2_effect_size_updated.png"

df = pd.read_csv(TABLE_PATH)

model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]

def cohens_d(m):
    a_mean = df[(df["model"] == m) & (df["group"] == "A")]["her_mean"].values[0]
    a_std = df[(df["model"] == m) & (df["group"] == "A")]["her_std"].values[0]
    c_mean = df[(df["model"] == m) & (df["group"] == "C")]["her_mean"].values[0]
    c_std = df[(df["model"] == m) & (df["group"] == "C")]["her_std"].values[0]
    pooled = np.sqrt((a_std**2 + c_std**2) / 2)
    return (c_mean - a_mean) / pooled if pooled > 0 else 0

d_vals = [cohens_d(m) for m in model_order]

x = np.arange(len(model_order))
colors = ['#ff9999' if d < 0 else '#99ff99' for d in d_vals]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, d_vals, color=colors, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
for bar, d in zip(bars, d_vals):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.05 if h >= 0 else h - 0.15,
            f'{d:.3f}', ha='center', va='bottom' if h >= 0 else 'top',
            fontsize=10, fontweight='bold')
ax.set_xlabel('Model')
ax.set_ylabel("Cohen's d (HER: C vs A)")
ax.set_title('Effect Size of Manual Cleaning on HER vs Model Size')
ax.set_xticks(x)
ax.set_xticklabels(model_order)
ax.set_ylim(-4, 1)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_PATH}")