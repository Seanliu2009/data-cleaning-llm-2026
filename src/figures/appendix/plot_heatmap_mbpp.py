"""
Heatmap: MBPP downstream pass@1 across 6 models x 6 groups.

Rows: 6 models. Columns: 6 cleaning groups.
Color and cell text encode mean pass@1 across seeds.

Output: FIGURES_DIR / "fig_heatmap_mbpp.png"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import EVAL_RESULTS_DIR, FIGURES_DIR

INPUT  = EVAL_RESULTS_DIR / "mbpp_pass1_summary.csv"
OUTPUT = FIGURES_DIR / "fig_heatmap_mbpp.png"

MODELS       = ["llama1b", "qwen1.5b", "llama3b", "qwen3b", "qwen7b", "llama8b"]
MODEL_LABELS = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
GROUPS       = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
METRIC       = "pass@1_processed"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})

df = pd.read_csv(INPUT)
pivot = df.groupby(["model", "group"])[METRIC].mean().unstack()
pivot = pivot.reindex(index=MODELS, columns=GROUPS)
data = pivot.values.astype(float)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
               vmin=np.nanmin(data), vmax=np.nanmax(data))

ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels(GROUPS)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODEL_LABELS)

for i in range(len(MODELS)):
    for j in range(len(GROUPS)):
        v = data[i, j]
        txt = "N/A" if np.isnan(v) else f"{v:.3f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

cb = plt.colorbar(im, ax=ax)
cb.set_label(METRIC)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
