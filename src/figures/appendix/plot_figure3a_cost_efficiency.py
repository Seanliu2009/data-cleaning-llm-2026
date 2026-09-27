"""
Figure 3a: Cost Efficiency (ES) for each cleaning strategy, small models.
ES = (HER_A - HER_strategy) / time_cost * 100
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import EVAL_RESULTS_DIR, FIGURES_DIR

INPUT  = EVAL_RESULTS_DIR / "squad_full_results_officialf1.csv"
OUTPUT = FIGURES_DIR / "figure3a_cost_efficiency.png"

TIME = {"B1": 0.45, "B2_3B": 4.95, "B2_8B": 6.00, "B2_70B": 11.50, "C": 4.67}
SMALL_MODELS = ["llama1b", "qwen1.5b", "llama3b", "qwen3b"]

squad = pd.read_csv(INPUT)

def group_mean(group):
    sub = squad[(squad["model"].isin(SMALL_MODELS)) & (squad["group"] == group)]
    return sub["HER"].mean()

strategies = ["B1", "B2_3B", "B2_8B", "B2_70B", "C"]
es_vals = []
for s in strategies:
    delta_her = group_mean("A") - group_mean(s)
    es_vals.append(delta_her / TIME[s] * 100)

labels = ["B1 (Rule)", "B2_3B", "B2_8B", "B2_70B", "C (Manual)"]
colors = ["#2ca02c", "#d62728", "#d62728", "#8b0000", "#ff7f0e"]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, es_vals, color=colors, edgecolor="black", linewidth=1.5)
ax.axhline(0, color="black", linewidth=1)
for bar, v in zip(bars, es_vals):
    h = bar.get_height()
    off = 0.3 if h >= 0 else -0.5
    ax.text(bar.get_x() + bar.get_width() / 2., h + off, f"{v:.2f}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=12, fontweight="bold")
ax.set_ylabel("Efficiency Score (ES) = (ΔHER / Cost) × 100", fontsize=11)
ax.set_title("Cost Efficiency (Small Models)", fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
