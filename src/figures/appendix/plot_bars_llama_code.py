"""
Bar chart: Code pass@1 - Llama family.
"""
import pandas as pd
from _bars_common import make_family_bars, FIGURES_DIR
from paths import EVAL_RESULTS_DIR

DATASET = "code"
METRIC = "pass@1_processed"
FAMILY = "Llama"
TITLE = "Code pass@1"
YLABEL = "pass@1"
OUT = FIGURES_DIR / "fig_bars_llama_code.png"

CSV_MAP = {
    "squad": EVAL_RESULTS_DIR / "all11_squad_results.csv",
    "nq":    EVAL_RESULTS_DIR / "all11_nq_results.csv",
    "code":  EVAL_RESULTS_DIR / "all11_code_pass1.csv",
}
df = pd.read_csv(CSV_MAP[DATASET])

make_family_bars(df, FAMILY, METRIC, OUT, f"{TITLE} - {FAMILY} family", YLABEL)
