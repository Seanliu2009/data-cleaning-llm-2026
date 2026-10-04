"""
Table 2b: NQ full statistics.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from paths import EVAL_RESULTS_DIR, TABLES_DIR

SQUAD_CSV = EVAL_RESULTS_DIR / "all11_squad_results.csv"
NQ_CSV    = EVAL_RESULTS_DIR / "all11_nq_results.csv"
CODE_CSV  = EVAL_RESULTS_DIR / "all11_code_pass1.csv"

MODEL_ORDER = ["qwen0.5b", "llama1b", "qwen1.5b", "gemma2b", "qwen3b",
               "llama3b", "phi3.8b", "qwen7b", "llama8b", "gemma9b", "phi14b"]

MODEL_LABELS = {
    "qwen0.5b": "Qwen-0.5B",
    "llama1b":  "Llama-1B",
    "qwen1.5b": "Qwen-1.5B",
    "gemma2b":  "Gemma-2B",
    "qwen3b":   "Qwen-3B",
    "llama3b":  "Llama-3B",
    "phi3.8b":  "Phi-3.8B",
    "qwen7b":   "Qwen-7B",
    "llama8b":  "Llama-8B",
    "gemma9b":  "Gemma-9B",
    "phi14b":   "Phi-14B",
}

GROUP_ORDER = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]
COMPARISONS = ["B1", "B2_3B", "B2_8B", "B2_70B", "C"]
N_COMP = len(COMPARISONS)
TIME_COST = {"B1": 0.45, "B2_3B": 4.95, "B2_8B": 6.00, "B2_70B": 11.50, "C": 4.67}


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def anova_one(df, model, metric):
    sub = df[df["model"] == model]
    arrays = [sub[sub["group"] == g][metric].dropna().values for g in GROUP_ORDER]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        return None
    F, p = stats.f_oneway(*arrays)
    df1 = len(arrays) - 1
    df2 = sum(len(a) for a in arrays) - len(arrays)
    eta2 = (F * df1) / (F * df1 + df2) if (F * df1 + df2) > 0 else 0.0
    return dict(F=float(F), df1=df1, df2=df2, p=float(p), eta_squared=float(eta2))


def paired_ttest(df, metric, model):
    sub = df[df["model"] == model].copy()
    sub["seed"] = sub["seed"].astype(int)
    a_ser = sub[sub["group"] == "A"].set_index("seed")[metric].sort_index()
    if len(a_ser) < 2:
        return []
    rows = []
    for group in COMPARISONS:
        g_ser = sub[sub["group"] == group].set_index("seed")[metric].sort_index()
        common = a_ser.index.intersection(g_ser.index)
        if len(common) < 2:
            continue
        a = a_ser.loc[common].values.astype(float)
        g = g_ser.loc[common].values.astype(float)
        if np.allclose(a, g):
            t_stat, p_val, d = 0.0, 1.0, 0.0
        else:
            t_stat, p_val = stats.ttest_rel(g, a)
            diff = g - a
            sd = diff.std(ddof=1)
            d = float(diff.mean() / sd) if sd > 0 else 0.0
        p_bonf = min(p_val * N_COMP, 1.0)
        rows.append({
            "model": MODEL_LABELS[model], "metric": metric,
            "comparison": f"{group}_vs_A",
            "mean_a": round(float(a.mean()), 4),
            "mean_group": round(float(g.mean()), 4),
            "diff": round(float((g - a).mean()), 4),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_val),
            "p_bonferroni": round(float(p_bonf), 6),
            "cohens_d": round(d, 4),
            "n_seeds": len(common),
            "is_significant": bool(p_val < 0.05),
            "is_significant_bonferroni": bool(p_bonf < 0.05),
        })
    return rows


df = pd.read_csv(NQ_CSV)
df["model_label"] = df["model"].map(MODEL_LABELS)
s = df.groupby(["model_label", "group"]).agg(
    f1_mean=("F1", "mean"), f1_std=("F1", "std"),
    n=("F1", "count")).round(4).reset_index()
s = s.rename(columns={"model_label": "model"})
s["_m"] = s["model"].apply(lambda x: list(MODEL_LABELS.values()).index(x))
s["_g"] = s["group"].apply(lambda x: GROUP_ORDER.index(x))
s = s.sort_values(["_m", "_g"]).drop(columns=["_m", "_g"]).reset_index(drop=True)
s.to_csv(TABLES_DIR / "table2b_full_statistics_NQ.csv", index=False)
print("[OK] table2b_full_statistics_NQ.csv")
