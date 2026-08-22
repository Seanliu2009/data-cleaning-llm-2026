"""
Generate Figure 3: Cost-effectiveness analysis.
(a) Efficiency Score (ES) bar chart.
(b) Quality-cost frontier with F1 loss annotations.
Reads data from table3_cost_effectiveness.csv.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import FIGURES_DIR, TABLES_DIR

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table3_cost_effectiveness.csv")

# Extract data for small model (Llama-3.2-3B)
strategies = df["strategy"].values
est_time = df["est_time_h"].values
small_delta_her = df["small_model_delta_her"].values
small_delta_f1 = df["small_model_delta_f1"].values

# ES = (ΔHER / Cost) × 100
es_values = (small_delta_her / est_time) * 100

# For the frontier: A (baseline) is (0, 0)
# B1, B2, C from the table
times = [0] + est_time.tolist()
d_her = [0] + small_delta_her.tolist()
f1_loss = [0] + np.abs(small_delta_f1).tolist()
labels = ["A (Baseline)"] + strategies.tolist()

# Frontier line: A -> B1 -> C (exclude B2)
front_times = [0, est_time[0], est_time[2]]      # A, B1, C
front_her = [0, small_delta_her[0], small_delta_her[2]]

# Colors
colors_bar = ['#2ca02c', '#d62728', '#ff7f0e']
colors_scatter = ['gray', '#2ca02c', 'red', '#ff7f0e']
markers = ['o', 'o', 'X', 'o']
sizes = [80, 120, 200, 120]

# ------------------------------------------------------------------
# 2. Create combined figure
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---------- (a) Cost Efficiency Bar Chart ----------
bars = ax1.bar(strategies, es_values, color=colors_bar, edgecolor='black', linewidth=1.2)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('Efficiency Score (ES) = (ΔHER / Cost) × 100', fontsize=11)
ax1.set_title('(a) Cost Efficiency (Small model)', fontsize=13, fontweight='bold')
ax1.set_ylim(-2.5, 5.0)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

for bar, val in zip(bars, es_values):
    y_pos = val + 0.15 if val > 0 else val - 0.15
    va = 'bottom' if val > 0 else 'top'
    ax1.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}',
             ha='center', va=va, fontweight='bold', fontsize=11, color='black')

# ---------- (b) Quality-Cost Frontier ----------
ax2.plot(front_times, front_her, 'b--', linewidth=2.5, color='#1f77b4',
         label='Quality-Cost Frontier', zorder=1)

for i, (x, y, label, col, marker, size) in enumerate(zip(times, d_her, labels,
                                                          colors_scatter, markers, sizes)):
    if label == 'B2 (LLM)':
        ax2.scatter(x, y, s=500, marker='X', color='red', edgecolor='black',
                    linewidth=2.5, zorder=5, label='B2 (Dominated)')
        ax2.annotate('B2 (Not Recommended)',
                     xy=(x, y), xytext=(x - 1.5, y - 0.025),
                     fontsize=10, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    else:
        ax2.scatter(x, y, s=size, color=col, edgecolor='black', linewidth=1.5,
                    zorder=3, marker=marker)
        label_clean = label.split()[0]
        ax2.annotate(label_clean, xy=(x, y), xytext=(x + 0.15, y + 0.003),
                     fontsize=11, fontweight='bold', color='black')

        if label_clean in ['B1', 'C']:
            loss_val = f1_loss[i]
            ax2.annotate(f'F1 Loss: {loss_val:.3f}',
                         xy=(x, y), xytext=(x + 0.15, y - 0.006),
                         fontsize=9, color='darkred', fontweight='normal',
                         arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
ax2.set_xlabel('Time Cost (hours)', fontsize=11)
ax2.set_ylabel('ΔHER (Reduction in Hallucination)', fontsize=11)
ax2.set_title('(b) Quality-Cost Frontier with F1 Penalty', fontsize=13, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.25)
ax2.set_xlim(-0.5, 6.5)
ax2.set_ylim(-0.10, 0.06)

ax2.text(4.2, 0.045, 'F1 Loss values are shown next to each point',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ------------------------------------------------------------------
# 3. Final adjustments
# ------------------------------------------------------------------
plt.suptitle('Figure 3: Cost-Effectiveness Analysis of Cleaning Strategies',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3_cost_efficiency.png", dpi=300, bbox_inches='tight')
print(f"[INFO] Figure 3 saved to {FIGURES_DIR / 'figure3_cost_efficiency.png'}")
plt.show()