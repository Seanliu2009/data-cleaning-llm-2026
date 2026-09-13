"""
Figure 3a: Cost efficiency (small model) - Efficiency Score (ES) for B1, B2, C.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.paths import TABLES_DIR, FIGURES_DIR

# Read table3 to get ES components, or compute from table2a
TABLE2A = TABLES_DIR / "table2a_full_statistics_squad.csv"
OUTPUT_PATH = FIGURES_DIR / "figure3a_cost_efficiency_updated.png"

df = pd.read_csv(TABLE2A)

small_models = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B"]

def avg_her(models, group):
    return df[(df["model"].isin(models)) & (df["group"] == group)]["her_mean"].mean()

delta_her = {
    "B1": avg_her(small_models, "A") - avg_her(small_models, "B1"),
    "B2": avg_her(small_models, "A") - avg_her(small_models, "B2"),
    "C":  avg_her(small_models, "A") - avg_her(small_models, "C"),
}
time_cost = {"B1": 0.45, "B2": 4.95, "C": 4.67}
es = {g: delta_her[g] / time_cost[g] * 100 for g in ["B1", "B2", "C"]}

labels = ["B1 (Rule)", "B2 (LLM)", "C (Manual)"]
values = [es["B1"], es["B2"], es["C"]]
colors = ['#2ca02c', '#d62728', '#ff7f0e']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
for bar, v in zip(bars, values):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3 if h >= 0 else h - 0.5,
            f'{v:.2f}', ha='center', va='bottom' if h >= 0 else 'top',
            fontsize=13, fontweight='bold')
ax.set_ylabel('Efficiency Score (ES) = (ΔHER / Cost) × 100', fontsize=12)
ax.set_title('Cost Efficiency (Small Model)', fontsize=14, fontweight='bold')
ax.set_ylim(min(values) - 2, max(values) + 3)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
print(f"Saved: {OUTPUT_PATH}")