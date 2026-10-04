"""
Figure 5b: Mis-modification rate (%) of B2 cleaners across datasets.

Layout:
  - y-axis ticks capped at 100 (no misleading ticks above 100)
  - ylim (80, 106) for label + legend headroom
  - legend anchored to the far upper-right
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import TABLES_DIR, FIGURES_DIR

TABLE6_CSV = TABLES_DIR / "table6_b2_cleaner_comparison.csv"
OUT = FIGURES_DIR / "figure5b_mis_mod_rate.png"

if not TABLE6_CSV.exists():
    raise FileNotFoundError(f"{TABLE6_CSV} not found")

t6 = pd.read_csv(TABLE6_CSV)
datasets = ["SQuAD", "NQ", "MBPP"]
cleaners = ["B2_3B", "B2_8B", "B2_70B"]
colors = ["#66b3ff", "#ff9999", "#99ff99"]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(datasets))
width = 0.25

for i, cl in enumerate(cleaners):
    vals = []
    for ds in datasets:
        row = t6[(t6["dataset"] == ds) & (t6["cleaner"] == cl)]
        vals.append(float(row["mis_mod_rate_%"].values[0]) if len(row) else 0.0)
    bars = ax.bar(x + (i - 1) * width, vals, width, label=cl,
                  color=colors[i], edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Dataset", fontsize=10)
ax.set_ylabel("Mis-modification rate (%)", fontsize=10)
ax.set_title("Mis-modification rate (%) of B2 cleaners",
             fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["SQuAD", "NQ", "MBPP"], fontsize=9)

ax.set_ylim(80, 106)
ax.set_yticks([80, 85, 90, 95, 100])

ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] {OUT}")
