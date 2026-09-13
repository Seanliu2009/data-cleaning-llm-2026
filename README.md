# How do different data cleaning strategies affect factual consistency and hallucination in LLMs?

This repository contains the complete code and experimental results for the paper:

**"How do different data-cleaning strategies affect Factual Consistency and hallucinations in LLMs?"**

We systematically compare multiple cleaning strategies across six LLMs of different scales, using SQuAD 2.0, NQ-Open, and MBPP as evaluation benchmarks.

---

## Key Finding

**The effectiveness of data cleaning depends on model scale:**

| Model Size | Effect |
| :--- | :--- |
| **Small models (1B-3B)** | Cleaning reduces hallucination rate (HER), but also decreases F1 score. The effect follows an inverted U-shape: weakest at 1B, strongest at 3B. |
| **Large models (7B-8B)** | Models are robust to semantic noise; cleaning is ineffective or even harmful. |

This pattern holds across extractive QA (SQuAD 2.0), open-domain QA (NQ-Open), and code generation (MBPP).

---

## Experimental Design

| Component | Details |
| :--- | :--- |
| **Models** | Llama-3.2-1B, Qwen-1.5B, Qwen-2.5-3B, Llama-3.2-3B, Qwen-2.5-7B, Llama-8B |
| **Datasets** | SQuAD 2.0 (main), NQ-Open (extension), MBPP (code generation extension) |
| **Cleaning Strategies** | A (No cleaning), B1 (Rule-based), B2 (LLM-assisted), C (Manual) |
| **B2 Cleaner Scales** | Llama-3.2-3B, Llama-8B, Llama-70B (cross-dataset comparison) |
| **Training** | QLoRA (4-bit, r=8, alpha=16) |
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
│ ├── analysis/ # Figure generation
│ └── tables/ # Table generation
├── figures/ # Generated figures (PNG)
└── tables/ # Generated summary tables (CSV)

text

---

## How to Reproduce

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
Set up paths
Edit config/paths.py to point to your local data and model directories.

Prepare data
Run the data preparation scripts in src/data_preparation/ for each dataset.

Train models
Run the training scripts in src/training/ for each model and cleaning strategy.

Evaluate models
Run the evaluation scripts in src/evaluation/.

Generate tables and figures
Run the analysis scripts in src/analysis/ and src/tables/.

Dependencies
Python 3.10+

PyTorch 2.0+

Transformers 4.36+

PEFT 0.7+

bitsandbytes

pandas, numpy, scipy, matplotlib, tqdm

See requirements.txt for the full list.

Citation
If you use this code or data in your research, please cite our paper:

text
@article{your_paper,
  title={How do different data cleaning strategies affect factual consistency and hallucination in LLMs?},
  author={Xiaoxiang Liu},
  journal={ACL},
  year={2026}
}
License
MIT