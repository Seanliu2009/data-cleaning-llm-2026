"""
Generate Table 3: Cost-effectiveness analysis of cleaning strategies.

Columns:
- strategy: cleaning strategy name
- est_time_h: estimated time cost in hours
- small_model_delta_her: ΔHER for small models (Llama-3.2-3B)
- small_model_delta_f1: ΔF1 for small models
- large_model_delta_her: ΔHER for large models (Llama-8B)
- large_model_delta_f1: ΔF1 for large models
- recommendation: practical recommendation
"""

import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import TABLES_DIR

data = [
    # Strategy, Time(h), Small ΔHER, Small ΔF1, Large ΔHER, Large ΔF1, Recommendation
    ("B1 (Rule-based)", 0.45, 0.017, -0.040, -0.015, -0.050, "Small models only"),
    ("B2 (LLM-assisted)", 4.95, -0.067, -0.084, -0.054, -0.078, "Not recommended"),
    ("C (Manual)", 4.67, 0.034, -0.056, -0.001, 0.001, "Small models only (fewer hallucinations but low F1)"),
]

df = pd.DataFrame(
    data,
    columns=[
        "strategy",
        "est_time_h",
        "small_model_delta_her",
        "small_model_delta_f1",
        "large_model_delta_her",
        "large_model_delta_f1",
        "recommendation",
    ],
)

df.to_csv(TABLES_DIR / "table3_cost_effectiveness.csv", index=False)
print(f"[INFO] Table 3 saved to {TABLES_DIR / 'table3_cost_effectiveness.csv'}")