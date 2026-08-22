"""
Generate Table 5: Paired t-test results for NQ-Open (A vs C, one-sided).

Columns:
- model: model name (lowercase, hyphen-separated)
- mean_a: mean F1 of group A across seeds
- std_a: standard deviation of group A
- mean_c: mean F1 of group C across seeds
- std_c: standard deviation of group C
- diff: mean difference (A - C)
- t_stat: t-statistic
- p_value: one-sided p-value (A > C)
- n_seeds: number of seeds used
- is_significant: True if p < 0.05
"""

import pandas as pd
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import TABLES_DIR

data = [
    # Model, mean_A, std_A, mean_C, std_C, diff, t_stat, p_value, n_seeds, significant
    ("qwen-1.5b", 0.080108, 0.011413, 0.039703, 0.001324, 0.040404, 7.3532, 0.000911, 5, True),
    ("llama-3.2-3b", 0.078224, 0.008836, 0.056327, 0.000259, 0.021897, 4.9032, 0.004013, 5, True),
    ("qwen-2.5-7b", 0.116607, 0.008732, 0.061656, 0.002079, 0.054951, 14.5310, 0.000065, 5, True),
    ("llama-8b", 0.046529, 0.002120, 0.042728, 0.001078, 0.003801, 3.0834, 0.018405, 5, True),
]

df = pd.DataFrame(
    data,
    columns=[
        "model",
        "mean_a",
        "std_a",
        "mean_c",
        "std_c",
        "diff",
        "t_stat",
        "p_value",
        "n_seeds",
        "is_significant",
    ],
)

df.to_csv(TABLES_DIR / "table5_nq_paired_ttest.csv", index=False)
print(f"[INFO] Table 5 saved to {TABLES_DIR / 'table5_nq_paired_ttest.csv'}")