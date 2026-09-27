"""
Table 8 (appendix): Cohen's d summary matrix.

Reads the t-test outputs from table4/5/6.
Long format written to CSV; wide matrices printed to stdout.
"""
import pandas as pd

from paths import TABLES_DIR

T4 = TABLES_DIR / "table4_nq_paired_ttest_results.csv"
T5 = TABLES_DIR / "table5_squad_paired_ttest_results.csv"
T6 = TABLES_DIR / "table6_mbpp_paired_ttest_results.csv"
OUTPUT = TABLES_DIR / "table8_cohens_d_matrix.csv"

MODEL_ORDER = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
COMPARISONS = ["B1_vs_A", "B2_3B_vs_A", "B2_8B_vs_A", "B2_70B_vs_A", "C_vs_A"]

rows = []
for path, ds, forced_metric in [
    (T4, "NQ", "F1"),
    (T5, "SQuAD", None),
    (T6, "MBPP", "pass@1_processed"),
]:
    if not path.exists():
        print(f"[SKIP] missing {path}")
        continue
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({
            "dataset": ds,
            "metric": forced_metric if forced_metric else r["metric"],
            "model": r["model"],
            "comparison": r["comparison"],
            "cohens_d": r["cohens_d"],
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"Saved (long): {OUTPUT}  rows={len(df)}")

for (ds, met), sub in df.groupby(["dataset", "metric"]):
    pivot = sub.pivot(index="model", columns="comparison", values="cohens_d")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=COMPARISONS)
    print()
    print("=" * 70)
    print(f"{ds} | {met} | Cohen's d (paired, vs A)")
    print("=" * 70)
    print(pivot.round(2).to_string())
