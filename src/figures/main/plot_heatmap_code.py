"""
Heatmap: Code pass@1 (MBPP->HumanEval).
"""
from _plot_common import load_data, make_heatmap, FIGURES_DIR

squad, nq, code = load_data()
DATASET = "code"
METRIC = "pass@1_processed"
TITLE = "Code pass@1 (MBPP->HumanEval)"
REVERSE = False
OUT = FIGURES_DIR / "fig_heatmap_code.png"

df = {"squad": squad, "nq": nq, "code": code}[DATASET]
make_heatmap(df, METRIC, OUT, TITLE, reverse=REVERSE)
