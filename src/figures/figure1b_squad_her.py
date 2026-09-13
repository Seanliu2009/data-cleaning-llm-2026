"""
Figure 1b: SQuAD 2.0 HER comparison (A vs C) for all six models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.paths import TABLES_DIR, FIGURES_DIR

TABLE_PATH = TABLES_DIR / "table2a_full_statistics_squad.csv"
OUTPUT_PATH = FIGURES_DIR / "figure1b_squad_her_updated.png"

df = pd.read_csv(TABLE_PATH)

model_order = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
her_a = [df[(df["model"] == m) & (df["group"] == "A")]["her_mean"].values[0] for m in model_order]
her_c = [df[(df["model"] == m) & (df["group"] == "C")]["her_mean"].values[0] for m in model_order]
std_a = [df[(df["model"] == m) & (df["group"] == "A")]["her_std"].values[0] for m in model_order]
std_c = [df[(df["model"] == m) & (df["group"] == "C")]["her_std"].values[0] for m in model_order]

x = np.arange(len(model_order))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, her_a, width, label='A (No Cleaning)', color='#ff9999', yerr=std_a, capsize=3)
ax.bar(x + width/2, her_c, width, label='C (Manual Cleaning)', color='#66b3ff', yerr=std_c, capsize=3)
ax.set_xlabel('Model')
ax.set_ylabel('Hallucination Error Rate (HER)')
ax.set_title('SQuAD 2.0 HER: A vs C')
ax.set_xticks(x)
ax.set_xticklabels(model_order)
ax.legend()
ax.set_ylim(0, 0.6)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_PATH}")