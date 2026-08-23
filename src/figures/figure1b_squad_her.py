"""
Generate Figure 1b: SQuAD 2.0 HER comparison (A vs C).
Models are ordered from smallest to largest.
Data source: table2a_full_statistics_squad.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ------------------------------------------------------------------
# 1. Add project root to path so config can be imported
# ------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import FIGURES_DIR, TABLES_DIR

# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------
df = pd.read_csv(TABLES_DIR / "table2a_full_statistics_squad.csv")

# Model order: smallest to largest
model_order = ["qwen-1.5b", "llama-3.2-3b", "qwen-2.5-7b", "llama-8b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["model", "group"]).reset_index(drop=True)

df_a = df[df["group"] == "A"]
df_c = df[df["group"] == "C"]

display_names = ["Qwen-1.5B", "Llama-3.2-3B", "Qwen-2.5-7B", "Llama-8B"]

her_a_mean = df_a["her_mean"].values
her_a_std = df_a["her_std"].values
her_c_mean = df_c["her_mean"].values
her_c_std = df_c["her_std"].values

# ------------------------------------------------------------------
# 3. Create figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(display_names))
width = 0.35

bars1 = ax.bar(x - width/2, her_a_mean, width, yerr=her_a_std, capsize=4,
               label='A (No Cleaning)', color='#ff9999')
bars2 = ax.bar(x + width/2, her_c_mean, width, yerr=her_c_std, capsize=4,
               label='C (Manual Cleaning)', color='#66b3ff')

ax.set_ylabel('Hallucination Rate (HER)', fontsize=12)
ax.set_title('SQuAD 2.0: Hallucination Rate (A vs C)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(0, 0.4)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.008,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.008,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure1b_squad_her.png", dpi=300)
print(f"[INFO] Figure 1b saved to {FIGURES_DIR / 'figure1b_squad_her.png'}")
plt.show()