"""
Bar chart: SQuAD 2.0 F1 - Llama family.
"""
import pandas as pd
from _bars_common import make_family_bars, FIGURES_DIR
from paths import EVAL_RESULTS_DIR

DATASET = "squad"
METRIC = "F1"
FAMILY = "Llama"
TITLE = "SQuAD 2.0 F1"
YLABEL = "F1"
OUT = FIGURES_DIR / "fig_bars_llama_squad_f1.png"

CSV_MAP = {
    "squad": EVAL_RESULTS_DIR / "all11_squad_results.csv",
    "nq":    EVAL_RESULTS_DIR / "all11_nq_results.csv",
    "code":  EVAL_RESULTS_DIR / "all11_code_pass1.csv",
}
df = pd.read_csv(CSV_MAP[DATASET])

make_family_bars(df, FAMILY, METRIC, OUT, f"{TITLE} - {FAMILY} family", YLABEL)
