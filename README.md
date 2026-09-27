# How do different data cleaning strategies affect factual consistency and hallucination in LLMs?

This repository contains the complete code and experimental results for the paper:

**"How do different data-cleaning strategies affect Factual Consistency and hallucinations in LLMs?"**

We systematically compare multiple cleaning strategies across six LLMs of different scales, using SQuAD 2.0, NQ-Open, and MBPP as evaluation benchmarks.

---

## Key Finding

**Data cleaning never helps in this setup.** The net benefit of cleaning is zero or negative across all six models and three datasets.

| Cleaning strategy | Effect on downstream performance |
| :--- | :--- |
| **A** (No cleaning) | Baseline |
| **B1** (Rule-based) | Consistently slightly harmful on SQuAD (6/6 models), roughly equal to A on NQ |
| **B2** (LLM-assisted, 3B/8B/70B) | Consistently and largely harmful across all models and datasets |
| **C** (Manual) | Statistically indistinguishable from A on 5/6 models |

The degree of harm depends on **which field the cleaner corrupts**, not on model scale:

| Damage location | Example | Downstream impact |
| :--- | :--- | :--- |
| Output field (target answer) | SQuAD B2_3B replaced outputs with raw answer objects | Catastrophic; not recoverable |
| Input field (context only) | NQ B2 rewrote questions but preserved answers | Mild |
| Surface format only | MBPP B2 wrapped code in markdown fences | Recoverable via post-processing |

B2 cleaners exhibit a structural defect: even with a 70B cleaner, the repair rate stays below 18% and the mis-modification rate stays above 82%, across all three datasets.

---

## Experimental Design

| Component | Details |
| :--- | :--- |
| **Models** | Llama-3.2-1B, Qwen-1.5B, Qwen-2.5-3B, Llama-3.2-3B, Qwen-2.5-7B, Meta-Llama-3-8B |
| **Datasets** | SQuAD 2.0 (main), NQ-Open (extension), MBPP (code generation extension) |
| **Cleaning Strategies** | A (No cleaning), B1 (Rule-based), B2_3B / B2_8B / B2_70B (LLM-assisted at three scales), C (Manual) |
| **Training** | QLoRA (4-bit NF4, r=8, alpha=16, 3 epochs, LR 2e-4) |
| **Seeds** | 10 (SQuAD), 5 (NQ), 5 (MBPP) |
| **Total Runs** | Over 1000 training and evaluation runs |

---

## Repository Structure
.
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│ └── paths.py # Unified path management (multi-environment)
├── src/
│ ├── data_preparation/ # Data extraction, noise injection, cleaning
│ │ ├── squad/
│ │ ├── nq/
│ │ └── mbpp/
│ ├── training/ # QLoRA fine-tuning scripts
│ │ ├── squad/
│ │ ├── nq/
│ │ └── code/
│ ├── evaluation/ # Evaluation scripts
│ │ ├── squad/
│ │ ├── nq/
│ │ └── code/
│ ├── figures/
│ │ ├── main/ # Main-text figure scripts
│ │ └── appendix/ # Appendix figure scripts
│ └── tables/
│ ├── main/ # Main-text table scripts
│ └── appendix/ # Appendix table scripts
├── figures/
│ ├── main/ # Generated main-text figures (PNG)
│ └── appendix/ # Generated appendix figures (PNG)
└── tables/
├── main/ # Generated main-text tables (CSV)
└── appendix/ # Generated appendix tables (CSV)

text

---

## Main Text vs Appendix

The paper separates main-text items from appendix items. Scripts and outputs mirror this split.

### Main text

| Paper item | Script | Output |
| :--- | :--- | :--- |
| Table 1: summary across datasets | `src/tables/main/table1_main_summary.py` | `tables/main/table1_main_summary.csv` |
| Table 2: ANOVA summary | `src/tables/main/table2_anova_summary.py` | `tables/main/table2_anova_summary.csv` |
| Figure 1a: SQuAD heatmap | `src/figures/main/figure1a_heatmap_squad.py` | `figures/main/figure1a_heatmap_squad.png` |
| Figure 1b: NQ heatmap | `src/figures/main/figure1b_heatmap_nq.py` | `figures/main/figure1b_heatmap_nq.png` |
| Figure 1c: MBPP heatmap | `src/figures/main/figure1c_heatmap_mbpp.py` | `figures/main/figure1c_heatmap_mbpp.png` |
| Figure 2: B2 cleaner repair / mis-mod | `src/figures/main/figure2_b2_cleaner.py` | `figures/main/figure2_b2_cleaner.png` |
| Figure 3: cost efficiency | `src/figures/main/figure3_cost_efficiency.py` | `figures/main/figure3_cost_efficiency.png` |

### Appendix

| Paper item | Script | Output |
| :--- | :--- | :--- |
| Table A1: SQuAD full statistics | `src/tables/appendix/tableA1_squad_stats.py` | `tables/appendix/tableA1_squad_stats.csv` |
| Table A2: NQ full statistics | `src/tables/appendix/tableA2_nq_stats.py` | `tables/appendix/tableA2_nq_stats.csv` |
| Table A3: MBPP full statistics | `src/tables/appendix/tableA3_mbpp_stats.py` | `tables/appendix/tableA3_mbpp_stats.csv` |
| Table A4: cost data | `src/tables/appendix/tableA4_cost.py` | `tables/appendix/tableA4_cost.csv` |
| Table A5: NQ paired t-test | `src/tables/appendix/tableA5_nq_ttest.py` | `tables/appendix/tableA5_nq_ttest.csv` |
| Table A6: SQuAD paired t-test | `src/tables/appendix/tableA6_squad_ttest.py` | `tables/appendix/tableA6_squad_ttest.csv` |
| Table A7: MBPP paired t-test | `src/tables/appendix/tableA7_mbpp_ttest.py` | `tables/appendix/tableA7_mbpp_ttest.csv` |
| Table A8: full ANOVA | `src/tables/appendix/tableA8_anova_full.py` | `tables/appendix/tableA8_anova_full.csv` |
| Table A9: Cohen's d matrix | `src/tables/appendix/tableA9_cohens_d.py` | `tables/appendix/tableA9_cohens_d.csv` |
| Table A10: B2 cleaner comparison | `src/tables/appendix/tableA10_b2_cleaner.py` | `tables/appendix/tableA10_b2_cleaner.csv` |
| Figure B1-B6: grouped bar charts | `src/figures/appendix/figureB1_*.py` ... | `figures/appendix/figureB1_*.png` ... |
| Figure B7: effect size | `src/figures/appendix/figureB7_effect_size.py` | `figures/appendix/figureB7_effect_size.png` |
| Figure B8: quality-cost frontier | `src/figures/appendix/figureB8_cost_frontier.py` | `figures/appendix/figureB8_cost_frontier.png` |

---

## How to Reproduce

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
Set up paths

Edit config/paths.py to point to your local data and model directories. The file auto-detects ModelScope, AutoDL, and local environments via hostname; you can also override with environment variables.

Prepare data

Run the data preparation scripts under src/data_preparation/ for each dataset.

Train models

Run the training scripts under src/training/ for each model and cleaning strategy.

Evaluate models

Run the evaluation scripts under src/evaluation/.

Generate tables and figures

bash
python src/tables/main/table1_main_summary.py
python src/figures/main/figure1a_heatmap_squad.py
# ...
Environment
Python 3.10+

PyTorch 2.0+ (ROCm 7.x tested on AMD MI300X)

Transformers 4.36+

PEFT 0.7+

bitsandbytes 0.49.1+ (ROCm-compatible build required)

pandas, numpy, scipy, matplotlib, tqdm

See requirements.txt for the full list.

Notes on Evaluation Format
All training and evaluation scripts use the same chat template and tokenization:

Training applies apply_chat_template(messages, tokenize=False) on the full three-turn conversation, and on the first two turns with add_generation_prompt=True to build the prompt.

Evaluation applies the same chat template with add_generation_prompt=True, and tokenizes with add_special_tokens=False.

For Llama models, this is required to avoid double BOS injection.

The assistant loss mask is computed by comparing prompt token length with full token length.

Format mismatches between training and evaluation systematically degrade F1 by a large margin. See src/evaluation/ for the verification scripts.

Citation
bibtex
@inproceedings{liu2027datacleaning,
  title={How do different data-cleaning strategies affect Factual Consistency and hallucinations in LLMs?},
  author={Liu, Xiaoxiang},
  booktitle={Proceedings of the 2027 International Conference on Computational Linguistics (COLING)},
  year={2027}
}
License
MIT