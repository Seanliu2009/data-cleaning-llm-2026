"""
Table 1 (main text): Compact summary across three datasets and six groups.

Each row is one dataset. Columns are mean +/- std across all 6 models x seeds.
Metrics: SQuAD -> F1, NQ -> F1, MBPP -> pass@1_processed.
"""
import pandas as pd

from paths import EVAL_RESULTS_DIR, TABLES_DIR

SQUAD_CSV = EVAL_RESULTS_DIR / "squad_full_results_officialf1.csv"
NQ_CSV    = EVAL_RESULTS_DIR / "nq_full_results_officialf1.csv"
MBPP_CSV  = EVAL_RESULTS_DIR / "mbpp_pass1_summary.csv"
OUTPUT    = TABLES_DIR / "table1_main_summary.csv"

GROUPS = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]


def load(path, metric_col, dataset_name):
    df = pd.read_csv(path)
    df["dataset"] = dataset_name
    return df.rename(columns={metric_col: "value"})[["dataset", "group", "value"]]


sq = load(SQUAD_CSV, "F1", "SQuAD")
nq = load(NQ_CSV, "F1", "NQ")
mb = load(MBPP_CSV, "pass@1_processed", "MBPP")
combined = pd.concat([sq, nq, mb], ignore_index=True)

rows = []
for ds in ["SQuAD", "NQ", "MBPP"]:
    row = {"dataset": ds}
    for g in GROUPS:
        vals = combined[(combined["dataset"] == ds) &
                        (combined["group"] == g)]["value"].dropna()
        row[g] = f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}" if len(vals) else ""
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(out.to_string(index=False))

print()
print("Delta vs A (mean across models and seeds):")
for ds in ["SQuAD", "NQ", "MBPP"]:
    a_mean = combined[(combined["dataset"] == ds) &
                      (combined["group"] == "A")]["value"].mean()
    print(f"  {ds}:")
    for g in GROUPS:
        if g == "A":
            continue
        g_mean = combined[(combined["dataset"] == ds) &
                          (combined["group"] == g)]["value"].mean()
        print(f"    {g:8s} delta = {g_mean - a_mean:+.4f}")
