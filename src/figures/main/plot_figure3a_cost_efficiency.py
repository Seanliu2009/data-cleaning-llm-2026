"""
Figure 3a: Cost efficiency (all 11 models average).
"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import load_data, FIGURES_DIR

squad, _, _ = load_data()
TIME = {"B1": 0.45, "B2_3B": 4.95, "B2_8B": 6.00, "B2_70B": 11.50, "C": 4.67}


def mean_of(group, metric):
    return squad[squad["group"] == group][metric].mean()


strategies = ["B1", "B2_3B", "B2_8B", "B2_70B", "C"]
es_vals = [(mean_of("A", "HER") - mean_of(s, "HER")) / TIME[s] * 100
           for s in strategies]

labels = ["B1 (Rule)", "B2_3B", "B2_8B", "B2_70B", "C (Manual)"]
colors = ["#2ca02c", "#d62728", "#d62728", "#8b0000", "#ff7f0e"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, es_vals, color=colors, edgecolor="black", linewidth=1.2)
ax.axhline(0, color="black", linewidth=1)
for bar, v in zip(bars, es_vals):
    h = bar.get_height()
    off = 0.05 if h >= 0 else -0.15
    ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{v:.2f}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=10, fontweight="bold")
ax.set_ylabel("Efficiency Score (ES) = (ΔHER / Cost) × 100")
ax.set_title("Cost efficiency (11 models average)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3a_cost_efficiency.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure3a_cost_efficiency.png")
