"""
Generate Figure 4: Cleaning strategy decision map.
Based on model size and time budget.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import FIGURES_DIR

# ------------------------------------------------------------------
# 1. Define decision boundaries
# ------------------------------------------------------------------
# Based on Table 3:
# - Small model (≤4B): B1 if budget < 4.67h, C if budget ≥ 4.67h
# - Large model (≥6.5B): A (No Cleaning)
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
        if p >= 6.5 or b < 0.45:
            Z[i, j] = 0  # A (No Cleaning)
        elif p <= 4.0 and b < 4.67:
            Z[i, j] = 1  # B1 (Rule)
        elif p <= 4.0 and b >= 4.67:
            Z[i, j] = 2  # C (Manual)
        else:
            Z[i, j] = 0  # Fallback: A

colors = ['#ff9999', '#66b3ff', '#99ff99']
ax.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.7)

# Decision boundaries
ax.axvline(x=4.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=4.67, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=0.45, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Strategy labels
ax.text(1.5, 2.5, 'B1 (Rule-based)', fontsize=12, fontweight='bold', ha='center', alpha=0.9)
ax.text(1.5, 5.3, 'C (Manual)', fontsize=12, fontweight='bold', ha='center', alpha=0.9)
ax.text(7.0, 3.0, 'A (No Cleaning)', fontsize=12, fontweight='bold', ha='center', alpha=0.9)
ax.text(5.0, 5.3, 'B2 (LLM) NOT Recommended', fontsize=10, color='red',
        fontweight='bold', ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Model Size (Billion Parameters)')
ax.set_ylabel('Time Budget (Hours)')
ax.set_title('Cleaning Strategy Decision Map')
ax.set_xlim(0.5, 10)
ax.set_ylim(0, 6)
ax.grid(True, linestyle='--', alpha=0.2)

# Mark model positions
models = ['Qwen-1.5B', 'Llama-3.2-3B', 'Qwen-2.5-7B', 'Llama-8B']
params = [1.5, 3.0, 7.0, 8.0]
for p, name in zip(params, models):
    ax.axvline(x=p, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(p, -0.3, name, fontsize=8, ha='center', rotation=45, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure4_decision_map.png", dpi=300)
print(f"[INFO] Figure 4 saved to {FIGURES_DIR / 'figure4_decision_map.png'}")
plt.show()