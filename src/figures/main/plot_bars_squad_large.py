"""
Grouped bar chart: SQuAD downstream F1 - Larger models (3B-8B).

3 models, 6 cleaning groups per model.
Error bars = std across seeds.

Output: FIGURES_DIR / "fig_bars_squad_large.png"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import EVAL_RESULTS_DIR, FIGURES_DIR

INPUT  = EVAL_RESULTS_DIR / "squad_full_results_officialf1.csv"
OUTPUT = FIGURES_DIR / "fig_bars_squad_large.png"

GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
METRIC = "F1"
MODELS = ['qwen3b', 'qwen7b', 'llama8b']
LABELS = {
    "llama1b":  "Llama-1B",
    "qwen1.5b": "Qwen-1.5B",
    "llama3b":  "Llama-3B",
    "qwen3b":   "Qwen-3B",
    "qwen7b":   "Qwen-7B",
    "llama8b":  "Llama-8B",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

df = pd.read_csv(INPUT)
stats = df.groupby(["model", "group"])[METRIC].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(6.5, 4.0))

n_models = len(MODELS)
width = 0.25
x = np.arange(len(GROUPS))

for k, m in enumerate(MODELS):
    means, stds = [], []
    for g in GROUPS:
        row = stats[(stats["model"] == m) & (stats["group"] == g)]
        if len(row) == 0:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(float(row["mean"].values[0]))
            stds.append(float(row["std"].values[0]))
    offset = (k - (n_models - 1) / 2.0) * width
    ax.bar(x + offset, means, width, label=LABELS[m],
           yerr=stds, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(GROUPS)
ax.set_ylabel(METRIC)
ax.set_title("SQuAD - Larger models (3B-8B)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
