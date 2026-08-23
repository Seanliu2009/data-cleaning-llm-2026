"""
Generate Figure 3b: Quality-Cost Frontier with F1 Penalty.
Data source: table3_cost_effectiveness_squad.csv
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ------------------------------------------------------------------
# 1. Add project root to path so config can be imported
# ------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import FIGURES_DIR, TABLES_DIR

# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table3_cost_effectiveness_squad.csv")

# Actual column names from the CSV
strategies = df["strategy"].values
est_time = df["est.time_h_"].values
small_delta_her = df["small_model_delta_her"].values
small_delta_f1_abs = df["small_model_delta_f1"].abs().values

# A (Baseline): (0, 0)
times = [0] + list(est_time)
d_her = [0] + list(small_delta_her)
f1_loss = [0] + list(small_delta_f1_abs)
labels = ["A (Baseline)"] + list(strategies)

# Frontier line: A -> B1 -> C (exclude B2)
front_times = [0, est_time[0], est_time[2]]
front_her = [0, small_delta_her[0], small_delta_her[2]]

# Colors and markers
colors = ['gray', '#2ca02c', 'red', '#ff7f0e']
markers = ['o', 'o', 'X', 'o']
sizes = [80, 120, 200, 120]

# ------------------------------------------------------------------
# 3. Create figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Draw frontier line (A -> B1 -> C)
ax.plot(front_times, front_her, 'b--', linewidth=2.5, color='#1f77b4',
        label='Quality-Cost Frontier', zorder=1)

for i, (x, y, label, col, marker, size) in enumerate(zip(times, d_her, labels,
                                                          colors, markers, sizes)):
    if label == 'B2 (LLM-assisted)':
        # Red X for B2 (Dominated)
        ax.scatter(x, y, s=500, marker='X', color='red', edgecolor='black',
                   linewidth=2.5, zorder=5, label='B2 (Dominated)')
        # B2 (Not Recommended) annotation - moved to upper left
        ax.annotate('B2 (Not Recommended)',
                    xy=(x, y), xytext=(x - 1.8, y + 0.01),
                    fontsize=10, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        # F1 Loss annotation - moved to lower left, separate from the first
        loss_val = f1_loss[i]
        ax.annotate(f'F1 Loss: {loss_val:.3f}',
                    xy=(x, y), xytext=(x - 1.8, y - 0.025),
                    fontsize=9, color='darkred', fontweight='normal',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))
    else:
        ax.scatter(x, y, s=size, color=col, edgecolor='black', linewidth=1.5,
                   zorder=3, marker=marker)
        # Label strategy name
        label_clean = label.split()[0]
        # Adjust annotation positions for clarity
        if label_clean == 'A':
            ax.annotate(label_clean, xy=(x, y), xytext=(x + 0.15, y + 0.003),
                        fontsize=11, fontweight='bold', color='black')
        elif label_clean == 'C':
            ax.annotate(label_clean, xy=(x, y), xytext=(x + 0.15, y + 0.003),
                        fontsize=11, fontweight='bold', color='black')
            # F1 Loss for C - adjusted to avoid overlap
            loss_val = f1_loss[i]
            ax.annotate(f'F1 Loss: {loss_val:.3f}',
                        xy=(x, y), xytext=(x + 0.15, y - 0.008),
                        fontsize=9, color='darkred', fontweight='normal',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))
        else:
            # B1 label
            ax.annotate(label_clean, xy=(x, y), xytext=(x + 0.15, y + 0.003),
                        fontsize=11, fontweight='bold', color='black')
            # F1 Loss for B1
            loss_val = f1_loss[i]
            ax.annotate(f'F1 Loss: {loss_val:.3f}',
                        xy=(x, y), xytext=(x + 0.15, y - 0.008),
                        fontsize=9, color='darkred', fontweight='normal',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))

# Reference line at Y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)

ax.set_xlabel('Time Cost (hours)', fontsize=11)
ax.set_ylabel('ΔHER (Reduction in Hallucination)', fontsize=11)
ax.set_title('Quality-Cost Frontier with F1 Penalty', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.25)
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.10, 0.06)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3b_quality_cost_frontier.png", dpi=300)
print(f"[INFO] Figure 3b saved to {FIGURES_DIR / 'figure3b_quality_cost_frontier.png'}")
plt.show()