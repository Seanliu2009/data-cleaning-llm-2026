# How do different data cleaning strategies affect factual consistency and hallucination in LLMs?

This repository contains the complete code and experimental results for the paper:

**"How do different data-cleaning strategies affect Factual Consistency and hallucinations in LLMs?"**

We systematically compare four cleaning strategies across four LLMs of different scales, using SQuAD 2.0 and NQ-Open as evaluation benchmarks.

---

## Key Finding

**The effectiveness of data cleaning depends on model scale:**

| Model Size | Effect |
| :--- | :--- |
| **Small models (≤3B)** | Cleaning reduces hallucination rate (HER), but also decreases F1 score. |
| **Large models (≥7B)** | Models are robust to semantic noise; cleaning is ineffective or even harmful. |

This pattern holds across both extractive QA (SQuAD 2.0) and open-domain QA (NQ-Open).

---

## Experimental Design

| Component | Details |
| :--- | :--- |
| **Models** | Qwen1.5B, Llama3.2-3B, Qwen7B, Llama8B |
| **Datasets** | SQuAD 2.0 (main), NQ-Open (extension) |
| **Cleaning Strategies** | A (No cleaning), B1 (Rule-based), B2 (LLM-assisted), C (Manual) |
| **Training** | QLoRA (4-bit, r=8, alpha=16) |
| **Seeds** | 10 (main) / 5 (extension) |
| **Total Runs** | 240 |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── paths.py                 # Unified path management
├── src/
│   ├── data_preparation/        # Data extraction, noise injection, cleaning
│   │   ├── squad/
│   │   └── nq/
│   ├── training/                # QLoRA fine-tuning scripts
│   │   ├── squad/
│   │   └── nq/
│   ├── evaluation/              # Evaluation scripts
│   │   ├── squad/
│   │   └── nq/
│   └── analysis/                # Statistics and figure generation
├── figures/                     # Generated figures (PNG)
└── tables/                      # Generated summary tables (CSV)
```

---

## How to Reproduce

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up paths**
   Edit `config/paths.py` to point to your local data and model directories.

3. **Prepare data**
   Run the data preparation scripts in `src/data_preparation/`.

4. **Train models**
   Run the training scripts in `src/training/` for each model and cleaning strategy.

5. **Evaluate models**
   Run the evaluation scripts in `src/evaluation/`.

6. **Generate tables and figures**
   Run the analysis scripts in `src/analysis/`.

---

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- bitsandbytes
- pandas, numpy, scipy, matplotlib, tqdm

See `requirements.txt` for the full list.

---

## Citation

If you use this code or data in your research, please cite our paper:

```
@article{your_paper,
  title={How do different data cleaning strategies affect factual consistency and hallucination in LLMs?},
  author={Xiaoxiang Liu},
  journal={ACL},
  year={2026}
}
```

---

## License

MIT