"""
Repair rate of B2 cleaners.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _plot_common import TABLE6_CSV, FIGURES_DIR

if not TABLE6_CSV.exists():
    raise FileNotFoundError(f"{TABLE6_CSV} not found")

t6 = pd.read_csv(TABLE6_CSV)
datasets = ["SQuAD", "NQ", "MBPP"]
cleaners = ["B2_3B", "B2_8B", "B2_70B"]
colors = ["#66b3ff", "#ff9999", "#99ff99"]
COL = "repair_rate_%"
YLAB = "Repair rate (%)"
FILENAME = "figure5a_repair_rate.png"
YLIM = None

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(datasets))
width = 0.25
for i, cl in enumerate(cleaners):
    vals = []
    for ds in datasets:
        row = t6[(t6["dataset"] == ds) & (t6["cleaner"] == cl)]
        vals.append(float(row[COL].values[0]) if len(row) else 0.0)
    bars = ax.bar(x + (i - 1) * width, vals, width, label=cl,
                  color=colors[i], edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Dataset"); ax.set_ylabel(YLAB)
ax.set_title(f"{YLAB} of B2 cleaners")
ax.set_xticks(x); ax.set_xticklabels(["SQuAD", "NQ", "MBPP"])
ax.legend(fontsize=8)
if YLIM != "None":
    ax.set_ylim(*YLIM)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / FILENAME, dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] {FILENAME}")
