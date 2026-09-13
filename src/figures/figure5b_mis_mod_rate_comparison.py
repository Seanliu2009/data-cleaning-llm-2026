"""
Figure: Mis-modification rate of B2 cleaners (3B, 8B, 70B) across SQuAD, NQ, and MBPP.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from config.paths import FIGURES_DIR

OUTPUT_PATH = FIGURES_DIR / "figure_mis_mod_rate_comparison.png"

datasets = ["SQuAD", "NQ", "MBPP"]
cleaners = ["B2_3B", "B2_8B", "B2_70B"]

mis_mod_data = {
    "SQuAD": [99.06, 99.47, 99.82],
    "NQ":    [99.18, 99.24, 99.53],
    "MBPP":  [92.32, 92.44, 97.00],
}

colors = ["#66b3ff", "#ff9999", "#99ff99"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
width = 0.25

for i, cleaner in enumerate(cleaners):
    vals = [mis_mod_data[d][i] for d in datasets]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=cleaner,
                  color=colors[i], edgecolor='black')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Dataset')
ax.set_ylabel('Mis-Modification Rate (%)')
ax.set_title('Mis-Modification Rate of B2 Cleaners Across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='lower right')
ax.set_ylim(80, 105)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
print(f"Saved: {OUTPUT_PATH}")