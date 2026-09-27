"""
Table 2 (main text): Compact ANOVA summary, one row per dataset.
Also writes the full per-model ANOVA table for the appendix.

Factor: cleaning group (6 levels).
Effect size: eta squared.
Metric per dataset: SQuAD F1, NQ F1, MBPP pass@1_processed.
"""
import pandas as pd
from scipy import stats

from paths import EVAL_RESULTS_DIR, TABLES_DIR

SQUAD_CSV = EVAL_RESULTS_DIR / "squad_full_results_officialf1.csv"
NQ_CSV    = EVAL_RESULTS_DIR / "nq_full_results_officialf1.csv"
MBPP_CSV  = EVAL_RESULTS_DIR / "mbpp_pass1_summary.csv"
OUTPUT    = TABLES_DIR / "table2_anova_summary.csv"
APPENDIX  = TABLES_DIR / "tableA7_anova_full.csv"

MODELS = ["llama1b", "qwen1.5b", "llama3b", "qwen3b", "qwen7b", "llama8b"]
GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]


def load(path, metric_col, dataset_name):
    df = pd.read_csv(path)
    df["dataset"] = dataset_name
    return df.rename(columns={metric_col: "value"})[
        ["dataset", "model", "group", "value"]]


combined = pd.concat([
    load(SQUAD_CSV, "F1", "SQuAD"),
    load(NQ_CSV, "F1", "NQ"),
    load(MBPP_CSV, "pass@1_processed", "MBPP"),
], ignore_index=True)

per_model = []
for ds in ["SQuAD", "NQ", "MBPP"]:
    for m in MODELS:
        sub = combined[(combined["dataset"] == ds) & (combined["model"] == m)]
        arrays = [sub[sub["group"] == g]["value"].dropna().values
                  for g in GROUPS]
        arrays = [a for a in arrays if len(a) >= 2]
        if len(arrays) < 2:
            continue
        F, p = stats.f_oneway(*arrays)
        df1 = len(arrays) - 1
        df2 = sum(len(a) for a in arrays) - len(arrays)
        eta2 = (F * df1) / (F * df1 + df2) if (F * df1 + df2) > 0 else 0.0
        per_model.append({"dataset": ds, "model": m, "F": float(F),
                          "p": float(p), "eta_squared": float(eta2),
                          "significant": p < 0.05})

pm = pd.DataFrame(per_model)

summary = []
for ds in ["SQuAD", "NQ", "MBPP"]:
    sub = pm[pm["dataset"] == ds]
    if sub.empty:
        continue
    summary.append({
        "dataset": ds,
        "n_models": len(sub),
        "mean_eta_squared": round(sub["eta_squared"].mean(), 4),
        "min_eta_squared": round(sub["eta_squared"].min(), 4),
        "max_eta_squared": round(sub["eta_squared"].max(), 4),
        "F_range": f"{sub['F'].min():.1f} - {sub['F'].max():.1f}",
        "p_max": f"{sub['p'].max():.2e}",
        "n_significant": int(sub["significant"].sum()),
    })

out = pd.DataFrame(summary)
out.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(out.to_string(index=False))

pm_out = pm.copy()
pm_out["F"] = pm_out["F"].round(4)
pm_out["p"] = pm_out["p"].apply(lambda x: f"{x:.2e}")
pm_out["eta_squared"] = pm_out["eta_squared"].round(4)
pm_out.to_csv(APPENDIX, index=False)
print(f"\nSaved (appendix full): {APPENDIX}  rows={len(pm_out)}")
