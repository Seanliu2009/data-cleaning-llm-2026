"""
Figure 5a: Repair rate of B2 cleaners across three datasets.
Reads from the table6 CSV produced by rebuild_table6_b2_cleaner.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import TABLES_DIR, FIGURES_DIR

INPUT  = TABLES_DIR / "table6_b2_cleaner_comparison.csv"
OUTPUT = FIGURES_DIR / "figure5a_repair_rate.png"

if not INPUT.exists():
    raise FileNotFoundError(f"{INPUT} not found. Run rebuild_table6_b2_cleaner.py first.")

t6 = pd.read_csv(INPUT)

datasets = ["SQuAD", "NQ", "MBPP"]
cleaners = ["B2_3B", "B2_8B", "B2_70B"]
colors = ["#66b3ff", "#ff9999", "#99ff99"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
width = 0.25

for i, cleaner in enumerate(cleaners):
    vals = []
    for ds in datasets:
        row = t6[(t6["dataset"] == ds) & (t6["cleaner"] == cleaner)]
        vals.append(float(row["repair_rate_%"].values[0]) if len(row) else 0.0)
    bars = ax.bar(x + (i - 1) * width, vals, width, label=cleaner,
                  color=colors[i], edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Dataset")
ax.set_ylabel("Repair Rate (%)")
ax.set_title("Repair Rate of B2 Cleaners Across Datasets")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
