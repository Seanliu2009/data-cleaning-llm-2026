"""
Figure 1c: NQ-Open F1 score comparison (A vs B1 vs B2 vs C) for all six models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.paths import TABLES_DIR, FIGURES_DIR

TABLE_PATH = TABLES_DIR / "table2b_full_statistics_NQ.csv"
OUTPUT_PATH = FIGURES_DIR / "figure1c_nq_updated.png"

df = pd.read_csv(TABLE_PATH)

model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
groups = ["A", "B1", "B2", "C"]
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

x = np.arange(len(model_order))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, g in enumerate(groups):
    vals = [df[(df["model"] == m) & (df["group"] == g)]["f1_mean"].values[0] for m in model_order]
    ax.bar(x + (i - 1.5) * width, vals, width, label=g, color=colors[i])

ax.set_xlabel('Model')
ax.set_ylabel('F1 Score')
ax.set_title('NQ-Open F1 Score: A vs B1 vs B2 vs C')
ax.set_xticks(x)
ax.set_xticklabels(model_order)
ax.legend()
ax.set_ylim(0, 0.14)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_PATH}")