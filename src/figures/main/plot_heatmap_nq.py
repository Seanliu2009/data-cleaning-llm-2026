"""
Heatmap: NQ-Open F1.
"""
from _plot_common import load_data, make_heatmap, FIGURES_DIR

squad, nq, code = load_data()
DATASET = "nq"
METRIC = "F1"
TITLE = "NQ-Open F1"
REVERSE = False
OUT = FIGURES_DIR / "fig_heatmap_nq.png"

df = {"squad": squad, "nq": nq, "code": code}[DATASET]
make_heatmap(df, METRIC, OUT, TITLE, reverse=REVERSE)
