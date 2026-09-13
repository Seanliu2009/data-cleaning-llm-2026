"""
Figure 3b: Quality-cost frontier with F1 penalty.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.paths import TABLES_DIR, FIGURES_DIR

TABLE2A = TABLES_DIR / "table2a_full_statistics_squad.csv"
OUTPUT_PATH = FIGURES_DIR / "figure3b_quality_cost_frontier_updated.png"

df = pd.read_csv(TABLE2A)
small_models = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B"]

def avg(models, group, metric):
    return df[(df["model"].isin(models)) & (df["group"] == group)][metric].mean()

delta_her = {
    "B1": avg(small_models, "A", "her_mean") - avg(small_models, "B1", "her_mean"),
    "B2": avg(small_models, "A", "her_mean") - avg(small_models, "B2", "her_mean"),
    "C":  avg(small_models, "A", "her_mean") - avg(small_models, "C", "her_mean"),
}
f1_loss = {
    "B1": avg(small_models, "A", "f1_mean") - avg(small_models, "B1", "f1_mean"),
    "B2": avg(small_models, "A", "f1_mean") - avg(small_models, "B2", "f1_mean"),
    "C":  avg(small_models, "A", "f1_mean") - avg(small_models, "C", "f1_mean"),
}
time_cost = {"B1": 0.45, "B2": 4.95, "C": 4.67}

fig, ax = plt.subplots(figsize=(11, 7))
points = {
    "A":  (-0.2, 0.0),
    "B1": (time_cost["B1"], delta_her["B1"]),
    "C":  (time_cost["C"] - 0.5, delta_her["C"]),
    "B2": (time_cost["B2"] + 0.5, delta_her["B2"]),
}
colors_map = {"A": "gray", "B1": "green", "B2": "red", "C": "orange"}

for name, (x, y) in points.items():
    if name == "B2":
        ax.scatter(x, y, s=400, c=colors_map[name], marker='X', edgecolors='black', linewidth=2, zorder=5)
    else:
        ax.scatter(x, y, s=250, c=colors_map[name], edgecolors='black', linewidth=1.5, zorder=5)

ax.plot([points["A"][0], points["B1"][0], points["C"][0]],
        [points["A"][1], points["B1"][1], points["C"][1]],
        linestyle='--', color='steelblue', linewidth=2, alpha=0.8)

label_offsets = {"A": (0, 15), "B1": (0, 15), "C": (0, -22), "B2": (0, 22)}
for name, (x, y) in points.items():
    dx, dy = label_offsets[name]
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha='center', fontsize=14, fontweight='bold')

f1_loss_offsets = {"B1": (30, -35), "C": (-10, -28), "B2": (10, -28)}
for name in ["B1", "C", "B2"]:
    x, y = points[name]
    dx, dy = f1_loss_offsets[name]
    ax.annotate(f"F1 Loss: {f1_loss[name]:.3f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha='left' if dx > 0 else 'right',
                fontsize=10, color='darkred')

ax.annotate("B2 (Not Recommended)", points["B2"], textcoords="offset points",
            xytext=(0, 60), ha='center', fontsize=13, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlabel('Time Cost (hours)', fontsize=13)
ax.set_ylabel('ΔHER (Reduction in Hallucination)', fontsize=13)
ax.set_title('Quality-Cost Frontier with F1 Penalty', fontsize=15, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(-0.8, 6.8)
ax.set_ylim(min(delta_her.values()) - 0.03, max(delta_her.values()) + 0.04)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
print(f"Saved: {OUTPUT_PATH}")