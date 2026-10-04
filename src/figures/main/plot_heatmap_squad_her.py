"""
Heatmap: SQuAD 2.0 HER (lower = better).
"""
from _plot_common import load_data, make_heatmap, FIGURES_DIR

squad, nq, code = load_data()
DATASET = "squad"
METRIC = "HER"
TITLE = "SQuAD 2.0 HER (lower = better)"
REVERSE = True
OUT = FIGURES_DIR / "fig_heatmap_squad_her.png"

df = {"squad": squad, "nq": nq, "code": code}[DATASET]
make_heatmap(df, METRIC, OUT, TITLE, reverse=REVERSE)
