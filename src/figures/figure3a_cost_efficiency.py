"""
Generate Figure 3a: Cost Efficiency (ES) bar chart.
Data source: table3_cost_effectiveness_squad.csv
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
# 2. Load data
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table3_cost_effectiveness_squad.csv")

# Actual column names from the CSV
strategies = df["strategy"].values
est_time = df["est.time_h_"].values
small_delta_her = df["small_model_delta_her"].values

# Calculate ES = (ΔHER / Cost) × 100
es_values = (small_delta_her / est_time) * 100

# Display names for the plot
display_names = ["B1 (Rule)", "B2 (LLM)", "C (Manual)"]
colors_bar = ['#2ca02c', '#d62728', '#ff7f0e']

# ------------------------------------------------------------------
# 3. Create figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(display_names, es_values, color=colors_bar, edgecolor='black', linewidth=1.2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Efficiency Score (ES) = (ΔHER / Cost) × 100', fontsize=11)
ax.set_title('Cost Efficiency (Small Model)', fontsize=13, fontweight='bold')
ax.set_ylim(-2.5, 5.0)
ax.grid(axis='y', linestyle='--', alpha=0.3)

for bar, val in zip(bars, es_values):
    y_pos = val + 0.15 if val > 0 else val - 0.15
    va = 'bottom' if val > 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}',
            ha='center', va=va, fontweight='bold', fontsize=11, color='black')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3a_cost_efficiency.png", dpi=300)
print(f"[INFO] Figure 3a saved to {FIGURES_DIR / 'figure3a_cost_efficiency.png'}")
plt.show()