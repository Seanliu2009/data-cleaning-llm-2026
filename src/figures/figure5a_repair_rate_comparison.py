"""
Figure: Repair rate of B2 cleaners (3B, 8B, 70B) across SQuAD, NQ, and MBPP.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from config.paths import FIGURES_DIR

OUTPUT_PATH = FIGURES_DIR / "figure_repair_rate_comparison.png"

datasets = ["SQuAD", "NQ", "MBPP"]
cleaners = ["B2_3B", "B2_8B", "B2_70B"]

repair_data = {
    "SQuAD": [0.00, 4.07, 9.49],
    "NQ":    [2.03, 13.22, 9.83],
    "MBPP":  [4.26, 13.48, 17.73],
}

colors = ["#66b3ff", "#ff9999", "#99ff99"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
width = 0.25

for i, cleaner in enumerate(cleaners):
    vals = [repair_data[d][i] for d in datasets]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=cleaner,
                  color=colors[i], edgecolor='black')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Dataset')
ax.set_ylabel('Repair Rate (%)')
ax.set_title('Repair Rate of B2 Cleaners Across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.set_ylim(0, 22)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
print(f"Saved: {OUTPUT_PATH}")