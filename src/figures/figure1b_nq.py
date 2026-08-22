"""
Generate Figure 1b: NQ-Open F1 comparison across four cleaning strategies.
Reads data from table2_nq_stats.csv.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import FIGURES_DIR, TABLES_DIR

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table2_nq_stats.csv")

# Model order: consistent with Figure 1a
model_order = ["llama-3.2-3b", "llama-8b", "qwen-2.5-7b", "qwen-1.5b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["model", "group"]).reset_index(drop=True)

# Strategy order and colors
groups = ["A", "B1", "B2", "C"]
colors = {"A": "#1f77b4", "B1": "#2ca02c", "B2": "#d62728", "C": "#ff7f0e"}

x = np.arange(len(model_order))
width = 0.2

# ------------------------------------------------------------------
# 2. Create figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for i, group in enumerate(groups):
    means = []
    stds = []
    for model in model_order:
        row = df[(df["model"] == model) & (df["group"] == group)]
        if not row.empty:
            means.append(row["f1_mean"].values[0])
            stds.append(row["f1_std"].values[0])
        else:
            means.append(0)
            stds.append(0)
    offset = (i - 1.5) * width
    ax.bar(
        x + offset,
        means,
        width,
        label=group,
        color=colors[group],
        yerr=stds,
        capsize=3,
        error_kw={"linewidth": 1, "ecolor": "black"},
    )

# ------------------------------------------------------------------
# 3. Style settings
# ------------------------------------------------------------------
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_title("(b) NQ-Open", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_order, fontsize=10, rotation=0)
ax.set_ylim(0, 0.2)  # NQ-specific F1 range
ax.legend(title="Cleaning Strategy", fontsize=10, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure1b_nq.png", dpi=300)
print(f"[INFO] Figure 1b saved to {FIGURES_DIR / 'figure1b_nq.png'}")
plt.show()