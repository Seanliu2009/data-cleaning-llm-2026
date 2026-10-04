"""
Bar chart: SQuAD 2.0 HER - Gemma family.
"""
import pandas as pd
from _bars_common import make_family_bars, FIGURES_DIR
from paths import EVAL_RESULTS_DIR

DATASET = "squad"
METRIC = "HER"
FAMILY = "Gemma"
TITLE = "SQuAD 2.0 HER"
YLABEL = "HER"
OUT = FIGURES_DIR / "fig_bars_gemma_squad_her.png"

CSV_MAP = {
    "squad": EVAL_RESULTS_DIR / "all11_squad_results.csv",
    "nq":    EVAL_RESULTS_DIR / "all11_nq_results.csv",
    "code":  EVAL_RESULTS_DIR / "all11_code_pass1.csv",
}
df = pd.read_csv(CSV_MAP[DATASET])

make_family_bars(df, FAMILY, METRIC, OUT, f"{TITLE} - {FAMILY} family", YLABEL)
