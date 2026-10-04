# How do different data-cleaning strategies affect Factual Consistency and hallucinations in LLMs?

This repository contains the code, data-preparation pipelines, training scripts,
evaluation scripts, and result tables/figures for a systematic study on how
different data-cleaning strategies affect factual consistency and hallucination
behavior in LLMs.

## Key Finding

**Data cleaning never helps in this setup.** The net benefit of cleaning is
zero or negative across all eleven models (0.5B-14B, four families) and three
datasets.

| Cleaning strategy | Effect on downstream performance |
|---|---|
| **A** (No cleaning) | Baseline |
| **B1** (Rule-based) | Consistently slightly harmful on SQuAD (11/11 models); roughly equal to A on NQ |
| **B2** (LLM-assisted, 3B/8B/70B) | Consistently and largely harmful on SQuAD; mildly harmful on NQ; harmful only at the surface level on code (recoverable) |
| **C** (Manual) | Statistically indistinguishable from A on 11/11 models |

The degree of harm depends on **which field the cleaner corrupts**, not on model
scale or family.

| Damage location | Example | Downstream impact |
|---|---|---|
| Output field (target answer) | SQuAD B2_3B replaced outputs with raw answer objects | Catastrophic; not recoverable |
| Input field (context only) | NQ B2 rewrote questions but preserved answers | Mild |
| Surface format only | Code B2 wrapped code in markdown fences | Recoverable via post-processing |

This pattern holds across extractive QA (SQuAD 2.0), open-domain QA (NQ-Open),
and code generation (MBPP -> HumanEval), across 11 models from four families
(Llama, Qwen, Phi, Gemma).

## Repository Structure

Top-level layout:

- `README.md` - this file
- `requirements.txt` - Python dependencies
- `.gitignore` - files excluded from version control
- `config/paths.py` - unified path management (ModelScope / AutoDL / local)
- `src/data_preparation/` - data extraction, noise injection, cleaning
- `src/training/` - QLoRA fine-tuning scripts
- `src/evaluation/` - evaluation scripts
- `figures/main/` - main-text figure scripts
- `figures/appendix/` - appendix figure scripts
- `tables/main/` - main-text table scripts
- `tables/appendix/` - appendix table scripts
- `outputs/figures/` - generated PNGs
- `outputs/tables/` - generated CSVs

Inside `src/data_preparation/`:

- `squad/` - SQuAD 2.0 preparation scripts
- `nq/` - NQ-Open preparation scripts
- `mbpp/` - MBPP preparation scripts

Inside `src/training/`:

- `squad/` - SQuAD 2.0 QLoRA training
- `nq/` - NQ-Open QLoRA training
- `code/` - code-generation QLoRA training

Inside `src/evaluation/`:

- `squad/` - SQuAD 2.0 evaluation
- `nq/` - NQ-Open evaluation
- `code/` - code-generation evaluation

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.6+ (ROCm 7.x or CUDA 12.x)
- A ROCm- or CUDA-compatible GPU
- `bitsandbytes >= 0.49.1` (ROCm-compatible wheel for AMD GPUs)

Core dependencies:

- `transformers`
- `peft`
- `datasets`
- `accelerate`
- `bitsandbytes>=0.49.1`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `modelscope` (for downloading base models from ModelScope)

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

## B2 Data Consistency

The B2 cleaner writes its outputs to `ablation_data/{task}/`. Downstream
training reads the **same files** via `SQUAD_TRAIN_DATA`, `NQ_TRAIN_DATA`, and
`CODE_TRAIN_DATA` in `config/paths.py`. This guarantees that cleaning output
and training input are never out of sync.

## Citation

If you use this code or results, please cite:
@inproceedings{liu2027datacleaning,
title = {How do different data-cleaning strategies affect Factual
Consistency and hallucinations in LLMs?},
author = {Liu, Xiaoxiang},
year = {2027}
}

## License

MIT