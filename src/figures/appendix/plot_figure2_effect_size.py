"""
Figure 2: Cohen's d (C vs A) on SQuAD HER.
"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import load_data, MODEL_ORDER, MODEL_INFO, FIGURES_DIR

squad, _, _ = load_data()


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
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, d_vals, color=colors, edgecolor="black")
ax.axhline(0, color="black", linewidth=1)
for idx, (bar, d) in enumerate(zip(bars, d_vals)):
    h = bar.get_height()
    if idx == 0 or idx == len(bars) - 1:
        if h >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h - 0.03, f"{d:.3f}",
                    ha="center", va="top", fontsize=8, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{d:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        off = 0.05 if h >= 0 else -0.10
        ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{d:.3f}",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8, fontweight="bold")
ax.set_xlabel("Model"); ax.set_ylabel("Cohen's d (HER: C vs A)")
ax.set_title("Effect size of manual cleaning on HER (11 models)")
ax.set_xticks(x)
ax.set_xticklabels([MODEL_INFO[m][1] for m in MODEL_ORDER], rotation=30, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure2_effect_size.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure2_effect_size.png")
