"""
Path configuration for the data-cleaning-llm-2026 project.

Supports ModelScope (DSW-AMD), AutoDL, and local environments.
The correct environment is auto-detected from hostname / cwd.

All B2 cleaner outputs and B2 downstream training data point to the SAME
files under ABLATION_*_DIR. This guarantees cleaning / training consistency.
"""

import os
import socket
from pathlib import Path

# ============================================================
# 1. Environment detection
# ============================================================
hostname = socket.gethostname()
cwd = os.getcwd()

if "autodl" in hostname.lower():
    ENV = "autodl"
elif "dsw" in hostname.lower() or "/mnt/workspace" in cwd:
    ENV = "modelscope"
else:
    ENV = "local"

print(f"[INFO] Running in {ENV} environment (hostname: {hostname})")

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# 2. Environment-specific base paths
# ============================================================
if ENV == "autodl":
    WORKSPACE = Path("/root/autodl-tmp")
    MODEL_BASE = WORKSPACE / "models"
    ADAPTER_BASE = WORKSPACE / "models_new"
    CODE_ADAPTER_BASE = WORKSPACE / "models_new"

    SQUAD_ALPACA_DIR = WORKSPACE / "llama_data"
    NQ_DATA_DIR = WORKSPACE / "nq_data"
    CODE_DATA_DIR = WORKSPACE / "code_data"

    SQUAD_EVAL_FILE = WORKSPACE / "llama_data" / "squad2.0_eval_2000.json"
    NQ_DEV_FILE = NQ_DATA_DIR / "dev_nq_2000.json"
    HUMANEVAL_PATH = CODE_DATA_DIR / "humaneval" / "HumanEval.jsonl.gz"

elif ENV == "modelscope":
    WORKSPACE = Path("/mnt/workspace")
    MODEL_BASE = WORKSPACE / "models"
    ADAPTER_BASE = WORKSPACE / "models_new"
    CODE_ADAPTER_BASE = WORKSPACE / "models_new"

    SQUAD_ALPACA_DIR = WORKSPACE / "llama_data"
    NQ_DATA_DIR = WORKSPACE / "nq_data"
    CODE_DATA_DIR = WORKSPACE / "code_data"

    SQUAD_EVAL_FILE = WORKSPACE / "llama_data" / "squad2.0_eval_2000.json"
    NQ_DEV_FILE = NQ_DATA_DIR / "dev_nq_2000.json"
    HUMANEVAL_PATH = CODE_DATA_DIR / "humaneval" / "HumanEval.jsonl.gz"

else:
    WORKSPACE = PROJECT_ROOT / "workspace"
    MODEL_BASE = PROJECT_ROOT / "models"
    ADAPTER_BASE = PROJECT_ROOT / "adapters"
    CODE_ADAPTER_BASE = PROJECT_ROOT / "adapters_new"

    SQUAD_ALPACA_DIR = PROJECT_ROOT / "data" / "squad" / "alpaca"
    NQ_DATA_DIR = PROJECT_ROOT / "data" / "nq"
    CODE_DATA_DIR = PROJECT_ROOT / "data" / "code"

    SQUAD_EVAL_FILE = PROJECT_ROOT / "data" / "squad" / "squad2.0_eval_2000.json"
    NQ_DEV_FILE = NQ_DATA_DIR / "dev_nq_2000.json"
    HUMANEVAL_PATH = CODE_DATA_DIR / "humaneval" / "HumanEval.jsonl.gz"

# ============================================================
# 3. Output directories
# ============================================================
OUTPUT_DIR = WORKSPACE / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"
PROGRESS_DIR = OUTPUT_DIR / "progress"

EVAL_RESULTS_DIR = WORKSPACE / "eval_results"

MAIN_TABLES_DIR = TABLES_DIR / "main"
MAIN_FIGURES_DIR = FIGURES_DIR / "main"
APPENDIX_TABLES_DIR = TABLES_DIR / "appendix"
APPENDIX_FIGURES_DIR = FIGURES_DIR / "appendix"

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, PROGRESS_DIR,
          EVAL_RESULTS_DIR, MAIN_TABLES_DIR, MAIN_FIGURES_DIR,
          APPENDIX_TABLES_DIR, APPENDIX_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. Base model paths (main experiment, 6 models, 1B-8B)
# ============================================================
LLAMA_1B_PATH = MODEL_BASE / "models" / "models" / "LLM-Research--Llama-3.2-1B-Instruct" / "snapshots" / "master"
QWEN_1_5B_PATH = MODEL_BASE / "models" / "models" / "Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "master"
QWEN_3B_PATH = MODEL_BASE / "models" / "models" / "Qwen--Qwen2.5-3B-Instruct" / "snapshots" / "master"
LLAMA_3_2_3B_PATH = MODEL_BASE / "models" / "models" / "LLM-Research--Llama-3.2-3B-Instruct" / "snapshots" / "master"
QWEN_7B_PATH = MODEL_BASE / "models" / "models" / "Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "master"
LLAMA_8B_PATH = MODEL_BASE / "models" / "models" / "LLM-Research--Meta-Llama-3-8B-Instruct" / "snapshots" / "master"

# ============================================================
# 5. Base model paths (extension, 5 models, 0.5B-14B)
# ============================================================
QWEN_0_5B_PATH  = MODEL_BASE / "models" / "models" / "Qwen2.5-0.5B"
GEMMA_2B_PATH   = MODEL_BASE / "models" / "models" / "gemma-2-2b-it"
PHI_4_MINI_PATH = MODEL_BASE / "models" / "models" / "Phi-4-mini-3.8B"
GEMMA_9B_PATH   = MODEL_BASE / "models" / "models" / "gemma-2-9b-it"

if ENV == "autodl":
    PHI_4_14B_PATH = WORKSPACE / "models" / "Phi-4-14B"
    LLAMA_70B_PATH = WORKSPACE / "models" / "llama-70b-awq"
else:
    PHI_4_14B_PATH = MODEL_BASE / "models" / "models" / "Phi-4-14B"
    LLAMA_70B_PATH = None

# ============================================================
# 6. Adapter directories (main experiment)
# ============================================================
LLAMA_1B_SQUAD_ADAPTER = ADAPTER_BASE / "llama1b_squad"
QWEN_1_5B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen1.5b_squad"
QWEN_3B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen3b_squad"
LLAMA_3_2_3B_SQUAD_ADAPTER = ADAPTER_BASE / "llama3b_squad"
QWEN_7B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen7b_squad"
LLAMA_8B_SQUAD_ADAPTER = ADAPTER_BASE / "llama8b_squad"

LLAMA_1B_NQ_ADAPTER = ADAPTER_BASE / "llama1b_nq"
QWEN_1_5B_NQ_ADAPTER = ADAPTER_BASE / "qwen1.5b_nq"
QWEN_3B_NQ_ADAPTER = ADAPTER_BASE / "qwen3b_nq"
LLAMA_3_2_3B_NQ_ADAPTER = ADAPTER_BASE / "llama3b_nq"
QWEN_7B_NQ_ADAPTER = ADAPTER_BASE / "qwen7b_nq"
LLAMA_8B_NQ_ADAPTER = ADAPTER_BASE / "llama8b_nq"

LLAMA_1B_CODE_ADAPTER = CODE_ADAPTER_BASE / "llama1b_code"
QWEN_1_5B_CODE_ADAPTER = CODE_ADAPTER_BASE / "qwen1.5b_code"
QWEN_3B_CODE_ADAPTER = CODE_ADAPTER_BASE / "qwen3b_code"
LLAMA_3_2_3B_CODE_ADAPTER = CODE_ADAPTER_BASE / "llama3b_code"
QWEN_7B_CODE_ADAPTER = CODE_ADAPTER_BASE / "qwen7b_code"
LLAMA_8B_CODE_ADAPTER = CODE_ADAPTER_BASE / "llama8b_code"

# ============================================================
# 6.5 Adapter directories (extension models)
# ============================================================
QWEN_0_5B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen0.5b_squad"
QWEN_0_5B_NQ_ADAPTER    = ADAPTER_BASE / "qwen0.5b_nq"
QWEN_0_5B_CODE_ADAPTER  = ADAPTER_BASE / "qwen0.5b_code"

GEMMA_2B_SQUAD_ADAPTER  = ADAPTER_BASE / "gemma2b_squad"
GEMMA_2B_NQ_ADAPTER     = ADAPTER_BASE / "gemma2b_nq"
GEMMA_2B_CODE_ADAPTER   = ADAPTER_BASE / "gemma2b_code"

PHI_4_MINI_SQUAD_ADAPTER = ADAPTER_BASE / "phi3.8b_squad"
PHI_4_MINI_NQ_ADAPTER    = ADAPTER_BASE / "phi3.8b_nq"
PHI_4_MINI_CODE_ADAPTER  = ADAPTER_BASE / "phi3.8b_code"

PHI_4_14B_SQUAD_ADAPTER  = ADAPTER_BASE / "phi14b_squad"
PHI_4_14B_NQ_ADAPTER     = ADAPTER_BASE / "phi14b_nq"
PHI_4_14B_CODE_ADAPTER   = ADAPTER_BASE / "phi14b_code"

GEMMA_9B_SQUAD_ADAPTER   = ADAPTER_BASE / "gemma9b_squad"
GEMMA_9B_NQ_ADAPTER      = ADAPTER_BASE / "gemma9b_nq"
GEMMA_9B_CODE_ADAPTER    = ADAPTER_BASE / "gemma9b_code"

# ============================================================
# 7. Legacy code eval samples dirs
# ============================================================
CODE_EVAL_SAMPLES_DIR = WORKSPACE / "code_eval_samples_v3"
CODE_EVAL_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 8. Ablation (B2 cleaner) data directories
# ============================================================
# B2 cleaner writes its outputs here; downstream training reads the SAME
# files. This guarantees cleaning / training consistency.
ABLATION_SQUAD_DIR = WORKSPACE / "ablation_data" / "squad"
ABLATION_NQ_DIR    = WORKSPACE / "ablation_data" / "nq"
ABLATION_MBPP_DIR  = WORKSPACE / "ablation_data" / "mbpp"

for d in [ABLATION_SQUAD_DIR, ABLATION_NQ_DIR, ABLATION_MBPP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------- Active B2 data (used for both cleaning and downstream training) --------
SQUAD_B2_3B_PATH  = ABLATION_SQUAD_DIR / "train_B2C1_3B.json"
SQUAD_B2_8B_PATH  = ABLATION_SQUAD_DIR / "train_B2C1_8B.json"
SQUAD_B2_70B_PATH = ABLATION_SQUAD_DIR / "train_B2C1_70B.json"

NQ_B2_3B_PATH  = ABLATION_NQ_DIR / "train_nq_B2C1_3B.json"
NQ_B2_8B_PATH  = ABLATION_NQ_DIR / "train_nq_B2C1_8B.json"
NQ_B2_70B_PATH = ABLATION_NQ_DIR / "train_nq_B2C1_70B.json"

CODE_B2_3B_PATH  = ABLATION_MBPP_DIR / "train_code_B2C1_3B.json"
CODE_B2_8B_PATH  = ABLATION_MBPP_DIR / "train_code_B2C1_8B.json"
CODE_B2_70B_PATH = ABLATION_MBPP_DIR / "train_code_B2C1_70B.json"

# -------- Legacy B2 data (based on B1-cleaned data; NOT used downstream) --------
SQUAD_B2_3B_LEGACY  = SQUAD_ALPACA_DIR / "train_B2_3B.json"
SQUAD_B2_8B_LEGACY  = SQUAD_ALPACA_DIR / "train_B2_8B.json"
SQUAD_B2_70B_LEGACY = SQUAD_ALPACA_DIR / "train_B2_70B.json"

NQ_B2_3B_LEGACY  = NQ_DATA_DIR / "train_nq_B2_3B.json"
NQ_B2_8B_LEGACY  = NQ_DATA_DIR / "train_nq_B2_8B.json"
NQ_B2_70B_LEGACY = NQ_DATA_DIR / "train_nq_B2_70B.json"

CODE_B2_3B_LEGACY  = CODE_DATA_DIR / "train_code_B2_3B.json"
CODE_B2_8B_LEGACY  = CODE_DATA_DIR / "train_code_B2_8B.json"
CODE_B2_70B_LEGACY = CODE_DATA_DIR / "train_code_B2_70B.json"

# ============================================================
# 9. Unified training data dictionaries (use these in train scripts)
# ============================================================
# Every training script should read from these dicts so that cleaning output
# and downstream training data are guaranteed to be the same files.
SQUAD_TRAIN_DATA = {
    "A":      SQUAD_ALPACA_DIR / "train_A.json",
    "B1":     SQUAD_ALPACA_DIR / "train_B1.json",
    "B2_3B":  SQUAD_B2_3B_PATH,
    "B2_8B":  SQUAD_B2_8B_PATH,
    "B2_70B": SQUAD_B2_70B_PATH,
    "C":      SQUAD_ALPACA_DIR / "train_C.json",
}

NQ_TRAIN_DATA = {
    "A":      NQ_DATA_DIR / "train_nq_A.json",
    "B1":     NQ_DATA_DIR / "train_nq_B1.json",
    "B2_3B":  NQ_B2_3B_PATH,
    "B2_8B":  NQ_B2_8B_PATH,
    "B2_70B": NQ_B2_70B_PATH,
    "C":      NQ_DATA_DIR / "train_nq_C.json",
}

CODE_TRAIN_DATA = {
    "A":      CODE_DATA_DIR / "train_code_A.json",
    "B1":     CODE_DATA_DIR / "train_code_B1.json",
    "B2_3B":  CODE_B2_3B_PATH,
    "B2_8B":  CODE_B2_8B_PATH,
    "B2_70B": CODE_B2_70B_PATH,
    "C":      CODE_DATA_DIR / "train_code_C.json",
}

# Legacy templates (kept for backward compatibility; prefer the dicts above)
SQUAD_DATA_TEMPLATE = SQUAD_ALPACA_DIR / "train_{group}.json"
NQ_DATA_TEMPLATE = NQ_DATA_DIR / "train_nq_{group}.json"
CODE_DATA_TEMPLATE = CODE_DATA_DIR / "train_code_{group}.json"

# ============================================================
# 10. MBPP legacy paths
# ============================================================
MBPP_RAW_DIR = CODE_DATA_DIR / "raw" / "mbpp"
MBPP_PROCESSED_DIR = CODE_DATA_DIR / "processed" / "mbpp"

MBPP_CLEAN_FILE = MBPP_PROCESSED_DIR / "train_code_clean.jsonl"
MBPP_A_FILE = MBPP_PROCESSED_DIR / "train_code_A.jsonl"
MBPP_B1_FILE = MBPP_PROCESSED_DIR / "train_code_B1.jsonl"
MBPP_B2_3B_FILE = MBPP_PROCESSED_DIR / "train_code_B2_3B.jsonl"
MBPP_B2_8B_FILE = MBPP_PROCESSED_DIR / "train_code_B2_8B.jsonl"
MBPP_B2_70B_FILE = MBPP_PROCESSED_DIR / "train_code_B2_70B.jsonl"
MBPP_C_FILE = MBPP_PROCESSED_DIR / "train_code_C.jsonl"
MBPP_CLEANING_SHEET = MBPP_PROCESSED_DIR / "manual_cleaning_sheet_mbpp.xlsx"
MBPP_NOISE_LOG = MBPP_PROCESSED_DIR / "modification_log_mbpp.json"

for d in [MBPP_RAW_DIR, MBPP_PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 11. Result output aliases
# ============================================================
SQUAD_RESULTS = EVAL_RESULTS_DIR
NQ_RESULTS = EVAL_RESULTS_DIR
CODE_RESULTS = EVAL_RESULTS_DIR

# ============================================================
# 12. System prompts (must match training instructions byte-for-byte)
# ============================================================
SQUAD_SYSTEM = ("Answer the question based on the given context. "
                "If the answer is not in the context, respond with 'I don't know.'")
NQ_SYSTEM = ("Answer the question based on your knowledge. "
             "If you don't know, respond with 'I don't know.'")

# ============================================================
# 13. Helper
# ============================================================
def print_paths():
    print("\n" + "=" * 62)
    print("Loaded paths")
    print("=" * 62)
    print(f"ENV                     : {ENV}")
    print(f"WORKSPACE               : {WORKSPACE}")
    print(f"MODEL_BASE              : {MODEL_BASE}")
    print(f"ADAPTER_BASE            : {ADAPTER_BASE}")
    print(f"SQUAD_ALPACA_DIR        : {SQUAD_ALPACA_DIR}")
    print(f"NQ_DATA_DIR             : {NQ_DATA_DIR}")
    print(f"CODE_DATA_DIR           : {CODE_DATA_DIR}")
    print(f"SQUAD_EVAL_FILE         : {SQUAD_EVAL_FILE}")
    print(f"NQ_DEV_FILE             : {NQ_DEV_FILE}")
    print(f"HUMANEVAL_PATH          : {HUMANEVAL_PATH}")
    print(f"EVAL_RESULTS_DIR        : {EVAL_RESULTS_DIR}")
    print(f"FIGURES_DIR             : {FIGURES_DIR}")
    print(f"TABLES_DIR              : {TABLES_DIR}")
    print("-" * 62)
    print("Active B2 data (cleaning output == training input):")
    print(f"SQUAD_B2_3B_PATH        : {SQUAD_B2_3B_PATH}")
    print(f"SQUAD_B2_8B_PATH        : {SQUAD_B2_8B_PATH}")
    print(f"SQUAD_B2_70B_PATH       : {SQUAD_B2_70B_PATH}")
    print(f"NQ_B2_3B_PATH           : {NQ_B2_3B_PATH}")
    print(f"CODE_B2_3B_PATH         : {CODE_B2_3B_PATH}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    print_paths()