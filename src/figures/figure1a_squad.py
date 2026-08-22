"""
Generate Figure 1a: SQuAD 2.0 F1 and HER comparison (A vs C).
Reads data from table1_squad_stats.csv.
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
df = pd.read_csv(TABLES_DIR / "table1_squad_stats.csv")

# Model order
model_order = ["llama-3.2-3b", "llama-8b", "qwen-2.5-7b", "qwen-1.5b"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values(["model", "group"]).reset_index(drop=True)

# Extract A and C groups
df_a = df[df["group"] == "A"]
df_c = df[df["group"] == "C"]

models = model_order

f1_a_mean = df_a["f1_mean"].values
f1_a_std  = df_a["f1_std"].values
f1_c_mean = df_c["f1_mean"].values
f1_c_std  = df_c["f1_std"].values

her_a_mean = df_a["her_mean"].values
her_a_std  = df_a["her_std"].values
her_c_mean = df_c["her_mean"].values
her_c_std  = df_c["her_std"].values

# ------------------------------------------------------------------
# 2. Create figure
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(models))
width = 0.35

# F1 subplot
ax = axes[0]
ax.bar(x - width/2, f1_a_mean, width, yerr=f1_a_std, capsize=4,
       label='A (No Cleaning)', color='#ff9999')
ax.bar(x + width/2, f1_c_mean, width, yerr=f1_c_std, capsize=4,
       label='C (Manual Cleaning)', color='#66b3ff')
ax.set_ylabel('F1 Score')
ax.set_title('F1: No Cleaning vs Manual Cleaning')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(0, 1)

# HER subplot
ax = axes[1]
ax.bar(x - width/2, her_a_mean, width, yerr=her_a_std, capsize=4,
       label='A (No Cleaning)', color='#ff9999')
ax.bar(x + width/2, her_c_mean, width, yerr=her_c_std, capsize=4,
       label='C (Manual Cleaning)', color='#66b3ff')
ax.set_ylabel('Hallucination Rate (HER)')
ax.set_title('HER: No Cleaning vs Manual Cleaning')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(0, 0.4)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure1a_squad.png", dpi=300)
print(f"[INFO] Figure 1a saved to {FIGURES_DIR / 'figure1a_squad.png'}")
plt.show()