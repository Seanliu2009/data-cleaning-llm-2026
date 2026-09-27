"""
Figure 3b: Quality-cost frontier with F1 penalty, small models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import EVAL_RESULTS_DIR, FIGURES_DIR

INPUT  = EVAL_RESULTS_DIR / "squad_full_results_officialf1.csv"
OUTPUT = FIGURES_DIR / "figure3b_quality_cost_frontier.png"

TIME = {"B1": 0.45, "B2_3B": 4.95, "B2_8B": 6.00, "B2_70B": 11.50, "C": 4.67}
SMALL_MODELS = ["llama1b", "qwen1.5b", "llama3b", "qwen3b"]

squad = pd.read_csv(INPUT)

def gm(group, metric):
    sub = squad[(squad["model"].isin(SMALL_MODELS)) & (squad["group"] == group)]
    return sub[metric].mean()

points = {}
for s in ["B1", "B2_3B", "B2_8B", "B2_70B", "C"]:
    delta_her = gm("A", "HER") - gm(s, "HER")
    delta_f1 = gm(s, "F1") - gm("A", "F1")
    points[s] = (TIME[s], delta_her, -delta_f1)

fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(-0.2, 0.0, s=250, c="gray", edgecolors="black", linewidth=1.5, zorder=5)
ax.annotate("A", (-0.2, 0.0), textcoords="offset points",
            xytext=(0, 15), ha="center", fontsize=14, fontweight="bold")

colors_map = {"B1": "green", "B2_3B": "red", "B2_8B": "darkred",
              "B2_70B": "darkred", "C": "orange"}
markers = {"B1": "o", "B2_3B": "X", "B2_8B": "X", "B2_70B": "X", "C": "o"}

for s, (x, y, f1_loss) in points.items():
    ax.scatter(x, y, s=300, c=colors_map[s], marker=markers[s],
               edgecolors="black", linewidth=2, zorder=5)
    ax.annotate(s, (x, y), textcoords="offset points",
                xytext=(0, 15), ha="center", fontsize=12, fontweight="bold")
    ax.annotate(f"F1 loss: {f1_loss:.3f}", (x, y),
                textcoords="offset points", xytext=(15, -25),
                ha="left", fontsize=9, color="darkred")

ax.plot([-0.2, TIME["B1"], TIME["C"]],
        [0.0, points["B1"][1], points["C"][1]],
        linestyle="--", color="steelblue", linewidth=2, alpha=0.7)

ax.set_xlabel("Time Cost (hours)", fontsize=12)
ax.set_ylabel("ΔHER (Reduction in Hallucination)", fontsize=12)
ax.set_title("Quality-Cost Frontier with F1 Penalty",
             fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
