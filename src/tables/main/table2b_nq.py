"""
Table 2b (appendix): NQ full statistics (F1) for 6 models x 6 groups.
"""
import pandas as pd

from paths import EVAL_RESULTS_DIR, TABLES_DIR

INPUT  = EVAL_RESULTS_DIR / "nq_full_results_officialf1.csv"
OUTPUT = TABLES_DIR / "table2b_full_statistics_NQ.csv"

MODEL_MAP = {
    "llama1b":  "Llama-1B",
    "qwen1.5b": "Qwen-1.5B",
    "llama3b":  "Llama-3B",
    "qwen3b":   "Qwen-3B",
    "qwen7b":   "Qwen-7B",
    "llama8b":  "Llama-8B",
}
MODEL_ORDER = ["Llama-1B", "Qwen-1.5B", "Llama-3B", "Qwen-3B", "Qwen-7B", "Llama-8B"]
GROUP_ORDER = ["A", "B1", "B2_3B", "B2_8B", "B2_70B", "C"]

df = pd.read_csv(INPUT)
df["model"] = df["model"].map(MODEL_MAP)

stats = df.groupby(["model", "group"]).agg(
    f1_mean=("F1", "mean"),
    f1_std=("F1", "std"),
    n=("F1", "count"),
).round(4).reset_index()

stats["m"] = stats["model"].apply(lambda x: MODEL_ORDER.index(x))
stats["g"] = stats["group"].apply(lambda x: GROUP_ORDER.index(x))
stats = stats.sort_values(["m", "g"]).drop(columns=["m", "g"])

stats.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(stats.to_string(index=False))
