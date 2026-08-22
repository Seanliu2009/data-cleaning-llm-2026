"""
Paired t-test for Llama-8B NQ evaluation results.

This script reads per-seed CSV files (each containing f1 scores for all samples)
and extracts the mean F1 for group A and group C across 5 seeds.
It then performs a one-sided paired t-test to test whether A > C.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import NQ_RESULTS

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
RESULT_DIR = NQ_RESULTS / "Llama-8B"
SEEDS = [42, 43, 44, 45, 46]          # 5 random seeds

# Storage for per-seed mean F1 values
a_vals = []
c_vals = []

print("Extracting per-seed F1 means for Llama-8B...")

for seed in SEEDS:
    # File names: Llama-8B_A_seed42.csv, Llama-8B_C_seed42.csv, etc.
    fname_a = RESULT_DIR / f"Llama-8B_A_seed{seed}.csv"
    fname_c = RESULT_DIR / f"Llama-8B_C_seed{seed}.csv"

    # Read the CSV, expect columns: question, prediction, gold_answer, f1
    df_a = pd.read_csv(fname_a)
    df_c = pd.read_csv(fname_c)

    # Average F1 over all samples in this seed
    f1_a = df_a['f1'].dropna().mean()
    f1_c = df_c['f1'].dropna().mean()

    a_vals.append(f1_a)
    c_vals.append(f1_c)
    print(f"seed {seed}: A = {f1_a:.6f}, C = {f1_c:.6f}")

# ------------------------------------------------------------------
# One-sided paired t-test: H0: A <= C, H1: A > C
# ------------------------------------------------------------------
t_stat, p_two_tailed = stats.ttest_rel(a_vals, c_vals)

# Convert to one-sided p-value (since t_stat > 0, we can just divide by 2)
p_one_tailed = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2

# ------------------------------------------------------------------
# Print results in a consistent format for Table 5
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("Llama-8B NQ Paired t-test Results (A vs C, one-sided)")
print("=" * 50)
print(f"n_seeds      = {len(a_vals)}")
print(f"mean_A       = {np.mean(a_vals):.6f}")
print(f"std_A        = {np.std(a_vals, ddof=1):.6f}")
print(f"mean_C       = {np.mean(c_vals):.6f}")
print(f"std_C        = {np.std(c_vals, ddof=1):.6f}")
print(f"diff (A-C)   = {np.mean(a_vals) - np.mean(c_vals):.6f}")
print(f"t_stat       = {t_stat:.4f}")
print(f"p_value      = {p_one_tailed:.6f}")
print(f"is_significant = {p_one_tailed < 0.05}")
print("=" * 50)