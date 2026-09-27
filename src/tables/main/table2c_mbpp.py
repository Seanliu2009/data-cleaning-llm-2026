"""
Table 2c (appendix): MBPP full statistics (pass@1_processed and pass@1_raw)
for 6 models x 6 groups.
"""
import pandas as pd

from paths import EVAL_RESULTS_DIR, TABLES_DIR

INPUT  = EVAL_RESULTS_DIR / "mbpp_pass1_summary.csv"
OUTPUT = TABLES_DIR / "table2c_full_statistics_MBPP.csv"

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
    pass1_processed_mean=("pass@1_processed", "mean"),
    pass1_processed_std=("pass@1_processed", "std"),
    pass1_raw_mean=("pass@1_raw", "mean"),
    pass1_raw_std=("pass@1_raw", "std"),
    n=("pass@1_processed", "count"),
).round(4).reset_index()

stats["m"] = stats["model"].apply(lambda x: MODEL_ORDER.index(x))
stats["g"] = stats["group"].apply(lambda x: GROUP_ORDER.index(x))
stats = stats.sort_values(["m", "g"]).drop(columns=["m", "g"])

stats.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(stats.to_string(index=False))
