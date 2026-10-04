"""
Heatmap: SQuAD 2.0 F1.
"""
from _plot_common import load_data, make_heatmap, FIGURES_DIR

squad, nq, code = load_data()
DATASET = "squad"
METRIC = "F1"
TITLE = "SQuAD 2.0 F1"
REVERSE = False
OUT = FIGURES_DIR / "fig_heatmap_squad_f1.png"

df = {"squad": squad, "nq": nq, "code": code}[DATASET]
make_heatmap(df, METRIC, OUT, TITLE, reverse=REVERSE)
