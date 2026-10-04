"""
Figure 3b: Quality-cost frontier.
"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import load_data, FIGURES_DIR

squad, _, _ = load_data()
TIME = {"B1": 0.45, "B2_3B": 4.95, "B2_8B": 6.00, "B2_70B": 11.50, "C": 4.67}


def mean_of(group, metric):
    return squad[squad["group"] == group][metric].mean()


strategies = ["B1", "B2_3B", "B2_8B", "B2_70B", "C"]
points = {}
for s in strategies:
    dh = mean_of("A", "HER") - mean_of(s, "HER")
    df1 = mean_of(s, "F1") - mean_of("A", "F1")
    points[s] = (TIME[s], dh, -df1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(-0.2, 0.0, s=250, c="gray", edgecolors="black", linewidth=1.5, zorder=5)
ax.annotate("A", (-0.2, 0.0), textcoords="offset points",
            xytext=(0, 15), ha="center", fontsize=12, fontweight="bold")
colors_map = {"B1": "green", "B2_3B": "red", "B2_8B": "darkred",
              "B2_70B": "darkred", "C": "orange"}
markers = {"B1": "o", "B2_3B": "X", "B2_8B": "X", "B2_70B": "X", "C": "o"}
for s, (xx, yy, f1_loss) in points.items():
    ax.scatter(xx, yy, s=300, c=colors_map[s], marker=markers[s],
               edgecolors="black", linewidth=2, zorder=5)
    ax.annotate(s, (xx, yy), textcoords="offset points",
                xytext=(0, 15), ha="center", fontsize=11, fontweight="bold")
    ax.annotate(f"F1 loss: {f1_loss:.3f}", (xx, yy),
                textcoords="offset points", xytext=(15, -25),
                ha="left", fontsize=8, color="darkred")
ax.plot([-0.2, TIME["B1"], TIME["C"]],
        [0.0, points["B1"][1], points["C"][1]],
        linestyle="--", color="steelblue", linewidth=2, alpha=0.7)
ax.set_xlabel("Time cost (hours)")
ax.set_ylabel("ΔHER (reduction in hallucination)")
ax.set_title("Quality-cost frontier (11 models average)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3b_quality_cost_frontier.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure3b_quality_cost_frontier.png")
