"""
Table 4 (appendix): Paired t-test on NQ, all 6 models.

Comparisons: B1 / B2_3B / B2_8B / B2_70B / C, each vs A (paired by seed).
Metric: F1.
Bonferroni correction: p * 5 within each model.
"""
import pandas as pd
import numpy as np
from scipy import stats

from paths import EVAL_RESULTS_DIR, TABLES_DIR

INPUT  = EVAL_RESULTS_DIR / "nq_full_results_officialf1.csv"
OUTPUT = TABLES_DIR / "table4_nq_paired_ttest_results.csv"

MODEL_MAP = {
    "llama1b":  "Llama-1B",
    "qwen1.5b": "Qwen-1.5B",
    "llama3b":  "Llama-3B",
    "qwen3b":   "Qwen-3B",
    "qwen7b":   "Qwen-7B",
    "llama8b":  "Llama-8B",
}
MODEL_ORDER = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
COMPARISONS = ["B1", "B2_3B", "B2_8B", "B2_70B", "C"]
N_COMPARISONS = len(COMPARISONS)

df = pd.read_csv(INPUT)
df["seed"] = df["seed"].astype(int)
df["model"] = df["model"].map(MODEL_MAP)

rows = []
for metric in ["F1"]:
    for m in MODEL_ORDER:
        sub_a = df[(df["model"] == m) & (df["group"] == "A")].sort_values("seed")
        if len(sub_a) < 2:
            continue
        for group in COMPARISONS:
            sub_g = df[(df["model"] == m) & (df["group"] == group)].sort_values("seed")
            if len(sub_g) < 2:
                continue
            merged = pd.merge(sub_a[["seed", metric]], sub_g[["seed", metric]],
                              on="seed", suffixes=("_a", "_g"))
            if len(merged) < 2:
                continue
            a_vals = merged[f"{metric}_a"].values.astype(float)
            g_vals = merged[f"{metric}_g"].values.astype(float)
            if np.allclose(a_vals, g_vals):
                t_stat, p_value, d = 0.0, 1.0, 0.0
            else:
                t_stat, p_value = stats.ttest_rel(g_vals, a_vals)
                diff = g_vals - a_vals
                sd = diff.std(ddof=1)
                d = float(diff.mean() / sd) if sd > 0 else 0.0
            p_bonf = min(p_value * N_COMPARISONS, 1.0)
            rows.append({
                "model": m,
                "metric": metric,
                "comparison": f"{group}_vs_A",
                "mean_a": round(a_vals.mean(), 4),
                "std_a":  round(a_vals.std(ddof=1), 4) if len(a_vals) > 1 else 0.0,
                "mean_group": round(g_vals.mean(), 4),
                "std_group":  round(g_vals.std(ddof=1), 4) if len(g_vals) > 1 else 0.0,
                "diff":   round(float((g_vals - a_vals).mean()), 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "p_bonferroni": round(float(p_bonf), 6),
                "cohens_d": round(d, 4),
                "n_seeds": len(merged),
                "is_significant": bool(p_value < 0.05),
                "is_significant_bonferroni": bool(p_bonf < 0.05),
            })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}  rows={len(out)}")
print(out.to_string(index=False))
