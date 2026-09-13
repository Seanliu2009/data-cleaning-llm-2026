"""
Figure 4: Cleaning strategy decision map.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
from config.paths import FIGURES_DIR

OUTPUT_PATH = FIGURES_DIR / "figure4_decision_map_updated.png"

fig, ax = plt.subplots(figsize=(12, 7))

ax.fill_between([0, 3], [0, 0], [0.45, 0.45], color='lightcoral', alpha=0.6, label='A (No Cleaning)')
ax.fill_between([0, 3], [0.45, 0.45], [4.67, 4.67], color='lightskyblue', alpha=0.6, label='B1 (Rule-based)')
ax.fill_between([0, 3], [4.67, 4.67], [6, 6], color='lightgreen', alpha=0.6, label='C (Manual)')
ax.fill_between([7, 10], [0, 0], [6, 6], color='lightcoral', alpha=0.6)
ax.fill_between([3, 7], [0, 0], [6, 6], color='lightgray', alpha=0.3)

ax.axvline(3, color='black', linestyle='--', linewidth=1.5)
ax.axvline(7, color='black', linestyle='--', linewidth=1.5)
ax.axhline(0.45, color='black', linestyle='--', linewidth=1)
ax.axhline(4.67, color='black', linestyle='--', linewidth=1)

model_positions = {
    "Llama-1B": (1.0, 0.15), "Qwen-1.5B": (1.5, 0.15),
    "Llama-3.2-3B": (3.0, 0.15), "Qwen-2.5-3B": (3.0, 0.35),
    "Qwen-2.5-7B": (7.0, 0.15), "Llama-8B": (8.0, 0.15),
}
for name, (x, y) in model_positions.items():
    ax.text(x, y, name, rotation=45, ha='right', va='bottom', fontsize=9, color='black')

ax.text(5, 5.5, "B2 (LLM) NOT Recommended", fontsize=14, fontweight='bold',
        color='red', ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
ax.text(1.5, 2.5, "Small Models", fontsize=14, fontweight='bold', ha='center')
ax.text(8.5, 2.5, "Large Models", fontsize=14, fontweight='bold', ha='center')

ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
ax.set_ylabel('Time Budget (Hours)', fontsize=12)
ax.set_title('Cleaning Strategy Decision Map', fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
print(f"Saved: {OUTPUT_PATH}")