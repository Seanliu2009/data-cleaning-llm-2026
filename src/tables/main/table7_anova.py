"""
Table 7 (appendix): One-way ANOVA on group factor, one row per dataset x model.

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
OUTPUT    = TABLES_DIR / "table7_anova_results.csv"

MODEL_ORDER = ["llama1b", "qwen1.5b", "llama3b", "qwen3b", "qwen7b", "llama8b"]
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

rows = []
for (ds, m), sub in combined.groupby(["dataset", "model"]):
    arrays = [sub[sub["group"] == g]["value"].dropna().values for g in GROUPS]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        continue
    F, p = stats.f_oneway(*arrays)
    df1 = len(arrays) - 1
    df2 = sum(len(a) for a in arrays) - len(arrays)
    eta2 = (F * df1) / (F * df1 + df2) if (F * df1 + df2) > 0 else 0.0
    rows.append({
        "dataset": ds,
        "model": m,
        "F": round(float(F), 4),
        "df1": df1,
        "df2": df2,
        "p_value": f"{p:.2e}",
        "eta_squared": round(float(eta2), 4),
        "is_significant": bool(p < 0.05),
    })

out = pd.DataFrame(rows)
out["ds_order"] = out["dataset"].map({"SQuAD": 0, "NQ": 1, "MBPP": 2})
out["m_order"] = out["model"].apply(lambda x: MODEL_ORDER.index(x))
out = out.sort_values(["ds_order", "m_order"]).drop(columns=["ds_order", "m_order"])

out.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}  rows={len(out)}")
print(out.to_string(index=False))
