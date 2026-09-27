"""
Figure 2: Cohen's d of manual cleaning (C vs A) on HER, per model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import EVAL_RESULTS_DIR, FIGURES_DIR

INPUT  = EVAL_RESULTS_DIR / "squad_full_results_officialf1.csv"
OUTPUT = FIGURES_DIR / "figure2_effect_size.png"

MODEL_ORDER  = ["llama1b", "qwen1.5b", "llama3b", "qwen3b", "qwen7b", "llama8b"]
MODEL_LABELS = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]

squad = pd.read_csv(INPUT)


def paired_cohens_d(a, b):
    diff = np.asarray(b) - np.asarray(a)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


d_vals = []
for m in MODEL_ORDER:
    a = squad[(squad["model"] == m) & (squad["group"] == "A")].sort_values("seed")["HER"].values
    c = squad[(squad["model"] == m) & (squad["group"] == "C")].sort_values("seed")["HER"].values
    n = min(len(a), len(c))
    d_vals.append(paired_cohens_d(a[:n], c[:n]))

x = np.arange(len(MODEL_ORDER))
colors = ["#ff9999" if d < 0 else "#99ff99" for d in d_vals]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, d_vals, color=colors, edgecolor="black")
ax.axhline(0, color="black", linewidth=1)
for bar, d in zip(bars, d_vals):
    h = bar.get_height()
    off = 0.05 if h >= 0 else -0.10
    ax.text(bar.get_x() + bar.get_width() / 2., h + off, f"{d:.3f}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=10, fontweight="bold")
ax.set_xlabel("Model")
ax.set_ylabel("Cohen's d (HER: C vs A)")
ax.set_title("Effect Size of Manual Cleaning on HER vs Model Size")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
