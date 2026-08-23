"""
Generate Figure 4: Cleaning Strategy Decision Map.
Based on model size and time budget.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from matplotlib.patches import Patch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import FIGURES_DIR

# ------------------------------------------------------------------
# 1. Define decision boundaries
# ------------------------------------------------------------------
# Decision rules:
# - Small model (≤3B): B1 if budget < 4.67h, C if budget ≥ 4.67h
# - Large model (>3B): A (No Cleaning)
# - B2 is NOT recommended everywhere

fig, ax = plt.subplots(figsize=(10, 6))

model_sizes = np.linspace(0.5, 10, 200)
budgets = np.linspace(0, 6, 200)
X, Y = np.meshgrid(model_sizes, budgets)

Z = np.zeros_like(X)
for i in range(len(budgets)):
    for j in range(len(model_sizes)):
        p = model_sizes[j]
        b = budgets[i]
        if p > 3.0 or b < 0.45:
            Z[i, j] = 0  # A (No Cleaning) - Red
        elif p <= 3.0 and b < 4.67:
            Z[i, j] = 1  # B1 (Rule) - Blue
        elif p <= 3.0 and b >= 4.67:
            Z[i, j] = 2  # C (Manual) - Green
        else:
            Z[i, j] = 0

# Color mapping: A=red, B1=blue, C=green
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.7)

# Decision boundaries
ax.axvline(x=3.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=4.67, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=0.45, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# B2 label - kept inside the figure
ax.text(4.5, 5.3, 'B2 (LLM) NOT Recommended', fontsize=10, color='red',
        fontweight='bold', ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Small/Large model labels - vertical orientation
ax.text(1.5, -0.6, 'Small Models', fontsize=11, fontweight='bold',
        ha='center', va='top', rotation=0, color='black')
ax.text(6.5, -0.6, 'Large Models', fontsize=11, fontweight='bold',
        ha='center', va='top', rotation=0, color='black')

ax.set_xlabel('Model Size (Billion Parameters)', fontsize=11)
ax.set_ylabel('Time Budget (Hours)', fontsize=11)
ax.set_title('Cleaning Strategy Decision Map', fontsize=13, fontweight='bold')
ax.set_xlim(0.5, 10)
ax.set_ylim(0, 6)
ax.grid(True, linestyle='--', alpha=0.2)

# Mark model positions
models = ['Qwen-1.5B', 'Llama-3.2-3B', 'Qwen-2.5-7B', 'Llama-8B']
params = [1.5, 3.0, 7.0, 8.0]
for p, name in zip(params, models):
    ax.axvline(x=p, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(p, -0.3, name, fontsize=8, ha='center', rotation=45, color='gray')

# ------------------------------------------------------------------
# 2. Add legend (A, B1, C) on the side
# ------------------------------------------------------------------
legend_elements = [
    Patch(facecolor='#ff9999', edgecolor='black', label='A (No Cleaning)'),
    Patch(facecolor='#66b3ff', edgecolor='black', label='B1 (Rule-based)'),
    Patch(facecolor='#99ff99', edgecolor='black', label='C (Manual)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
          title='Cleaning Strategy', title_fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure4_decision_map.png", dpi=300)
print(f"[INFO] Figure 4 saved to {FIGURES_DIR / 'figure4_decision_map.png'}")
plt.show()