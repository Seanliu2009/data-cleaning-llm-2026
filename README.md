# How do different data-cleaning strategies affect Factual Consistency and hallucinations in LLMs?

This repository contains the code, data-preparation pipelines, training scripts,
evaluation scripts, and result tables/figures for a systematic study on how
different data-cleaning strategies affect factual consistency and hallucination
behavior in LLMs of varying sizes.

## Key Finding

**Data cleaning never helps in this setup.** The net benefit of cleaning is
zero or negative across all six models and three datasets.
| Cleaning strategy | Effect on downstream performance |
|---|---|
| **A** (No cleaning) | Baseline |
| **B1** (Rule-based) | Consistently slightly harmful on SQuAD (6/6 models); roughly equal to A on NQ |
| **B2** (LLM-assisted, 3B/8B/70B) | Consistently and largely harmful across all models and datasets |
| **C** (Manual) | Statistically indistinguishable from A on 5/6 models |
The degree of harm depends on **which field the cleaner corrupts**, not on
model scale:

| Damage location | Example | Downstream impact |
|---|---|---|
| Output field (target answer) | SQuAD B2_3B replaced outputs with raw answer objects | Catastrophic; not recoverable |
| Input field (context only) | NQ B2 rewrote questions but preserved answers | Mild |
| Surface format only | MBPP B2 wrapped code in markdown fences | Recoverable via post-processing |

This pattern holds across extractive QA (SQuAD 2.0), open-domain QA (NQ-Open),
and code generation (MBPP).
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
│ └── evaluation/ # Evaluation scripts
│ ├── squad/
│ ├── nq/
│ └── code/
├── figures/
│ ├── main/ # Main-text figure scripts
│ └── appendix/ # Appendix figure scripts
├── tables/
│ ├── main/ # Main-text table scripts
│ └── appendix/ # Appendix table scripts
└── outputs/
├── figures/
│ ├── main/ # Generated main-text figures (PNG)
│ └── appendix/ # Generated appendix figures (PNG)
└── tables/
├── main/ # Generated main-text tables (CSV)
└── appendix/ # Generated appendix tables (CSV)
## Setup

### Requirements

- Python 3.10+
- PyTorch 2.1+
- ROCm-compatible or CUDA-compatible GPU
- `bitsandbytes >= 0.49.1` (ROCm-compatible build required for AMD GPUs)

Core dependencies:

transformers
peft
datasets
accelerate
bitsandbytes>=0.49.1
pandas
numpy
scipy
matplotlib
tqdm

See `requirements.txt` for the full list.

### Installation

git clone https://github.com/Seanliu2009/data-cleaning-llm-2026.git
cd data-cleaning-llm-2026
pip install -r requirements.txt

Before running any script, verify that `config/paths.py` points to your local
data and model directories. Paths are environment-aware and support ModelScope,
AutoDL, and local setups.

## Notes on Evaluation Format

All training and evaluation scripts use the **same chat template and
tokenization**:

- Training applies `apply_chat_template(messages, tokenize=False)` on the full
  three-turn conversation, and on the first two turns with
  `add_generation_prompt=True` to build the prompt.
- Evaluation applies the same chat template with `add_generation_prompt=True`,
  and tokenizes with `add_special_tokens=False`.
- For Llama models, this is required to avoid double BOS injection.
- The assistant loss mask is computed by comparing prompt token length with
  full token length.

Format mismatches between training and evaluation systematically degrade F1 by
a large margin. See `src/evaluation/` for the verification scripts.
## Citation

@inproceedings{liu2027datacleaning,
  title     = {How do different data-cleaning strategies affect Factual
               Consistency and hallucinations in LLMs?},
  author    = {Liu, Xiaoxiang},
  booktitle = {Proceedings of the 2027 International Conference on
               Computational Linguistics (COLING)},
  year      = {2027}
}

## License

MIT