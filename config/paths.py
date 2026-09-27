"""
Path configuration supporting multiple environments (AutoDL, ModelScope, local).

All scripts import paths from this module.
The correct environment is automatically selected based on hostname or current working directory.
"""

import os
import socket
from pathlib import Path

# ============================================================
# 1. Detect current environment
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

# ============================================================
# 2. Project root
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# 3. Environment-specific base paths
# ============================================================
if ENV == "autodl":
    WORKSPACE = Path("/root/autodl-tmp")
    MODEL_BASE = WORKSPACE / "models"
    ADAPTER_BASE = WORKSPACE / "models"
    CODE_ADAPTER_BASE = WORKSPACE / "models"

    SQUAD_ALPACA_DIR = WORKSPACE / "llama_data"
    NQ_DATA_DIR = WORKSPACE / "nq_data"
    CODE_DATA_DIR = WORKSPACE / "code_data"

    SQUAD_EVAL_FILE = WORKSPACE / "squad2.0_eval_2000.json"
    NQ_DEV_FILE = NQ_DATA_DIR / "dev_nq_2000.json"
    HUMANEVAL_PATH = CODE_DATA_DIR / "humaneval" / "HumanEval.jsonl.gz"

elif ENV == "modelscope":
    WORKSPACE = Path("/mnt/workspace")
    MODEL_BASE = WORKSPACE / "models"
    # All QLoRA adapters (SQuAD, NQ, code) live under models_new on ModelScope.
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
# 4. Common output directories
# ============================================================
OUTPUT_DIR = WORKSPACE / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

# Unified evaluation output directory.
# SQuAD, NQ and code evaluation CSVs all write here.
EVAL_RESULTS_DIR = WORKSPACE / "eval_results"

# Main text vs appendix split.
# Scripts and generated outputs mirror the paper structure:
#   main/      items referenced in the main paper
#   appendix/  items referenced in the appendix
MAIN_TABLES_DIR = TABLES_DIR / "main"
MAIN_FIGURES_DIR = FIGURES_DIR / "main"
APPENDIX_TABLES_DIR = TABLES_DIR / "appendix"
APPENDIX_FIGURES_DIR = FIGURES_DIR / "appendix"

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, EVAL_RESULTS_DIR,
          MAIN_TABLES_DIR, MAIN_FIGURES_DIR,
          APPENDIX_TABLES_DIR, APPENDIX_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 5. Base model paths
# ============================================================
LLAMA_1B_PATH = MODEL_BASE / "models" / "models" / "LLM-Research--Llama-3.2-1B-Instruct" / "snapshots" / "master"
QWEN_1_5B_PATH = MODEL_BASE / "models" / "models" / "Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "master"
QWEN_3B_PATH = MODEL_BASE / "models" / "models" / "Qwen--Qwen2.5-3B-Instruct" / "snapshots" / "master"
LLAMA_3_2_3B_PATH = MODEL_BASE / "models" / "models" / "LLM-Research--Llama-3.2-3B-Instruct" / "snapshots" / "master"
QWEN_7B_PATH = MODEL_BASE / "models" / "models" / "Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "master"
LLAMA_8B_PATH = MODEL_BASE / "models" / "models" / "LLM-Research--Meta-Llama-3-8B-Instruct" / "snapshots" / "master"

if ENV == "autodl":
    LLAMA_70B_PATH = WORKSPACE / "models" / "llama-70b-awq"
else:
    LLAMA_70B_PATH = None

# ============================================================
# 6. Adapter directories
# ============================================================
# SQuAD adapters
LLAMA_1B_SQUAD_ADAPTER = ADAPTER_BASE / "llama1b_squad"
QWEN_1_5B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen1.5b_squad"
QWEN_3B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen3b_squad"
LLAMA_3_2_3B_SQUAD_ADAPTER = ADAPTER_BASE / "llama3b_squad"
QWEN_7B_SQUAD_ADAPTER = ADAPTER_BASE / "qwen7b_squad"
LLAMA_8B_SQUAD_ADAPTER = ADAPTER_BASE / "llama8b_squad"

# NQ adapters
LLAMA_1B_NQ_ADAPTER = ADAPTER_BASE / "llama1b_nq"
QWEN_1_5B_NQ_ADAPTER = ADAPTER_BASE / "qwen1.5b_nq"
QWEN_3B_NQ_ADAPTER = ADAPTER_BASE / "qwen3b_nq"
LLAMA_3_2_3B_NQ_ADAPTER = ADAPTER_BASE / "llama3b_nq"
QWEN_7B_NQ_ADAPTER = ADAPTER_BASE / "qwen7b_nq"
LLAMA_8B_NQ_ADAPTER = ADAPTER_BASE / "llama8b_nq"

# Code generation adapters
LLAMA_1B_CODE_ADAPTER = CODE_ADAPTER_BASE / "llama1b_code"
QWEN_1_5B_CODE_ADAPTER = CODE_ADAPTER_BASE / "qwen1.5b_code"
QWEN_3B_CODE_ADAPTER = CODE_ADAPTER_BASE / "qwen3b_code"
LLAMA_3_2_3B_CODE_ADAPTER = CODE_ADAPTER_BASE / "llama3b_code"
QWEN_7B_CODE_ADAPTER = CODE_ADAPTER_BASE / "qwen7b_code"
LLAMA_8B_CODE_ADAPTER = CODE_ADAPTER_BASE / "llama8b_code"

# Code evaluation samples directory
CODE_EVAL_SAMPLES_DIR = WORKSPACE / "code_eval_samples_v3"
CODE_EVAL_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 7. Training data templates
# ============================================================
SQUAD_DATA_TEMPLATE = SQUAD_ALPACA_DIR / "train_{group}.json"
NQ_DATA_TEMPLATE = NQ_DATA_DIR / "train_nq_{group}.json"
CODE_DATA_TEMPLATE = CODE_DATA_DIR / "train_code_{group}.json"

# ============================================================
# 7.5 B2 cleaner output paths
# ============================================================
SQUAD_B2_3B_PATH = SQUAD_ALPACA_DIR / "train_B2_3B.json"
SQUAD_B2_8B_PATH = SQUAD_ALPACA_DIR / "train_B2_8B.json"
SQUAD_B2_70B_PATH = SQUAD_ALPACA_DIR / "train_B2_70B.json"

NQ_B2_3B_PATH = NQ_DATA_DIR / "train_nq_B2_3B.json"
NQ_B2_8B_PATH = NQ_DATA_DIR / "train_nq_B2_8B.json"
NQ_B2_70B_PATH = NQ_DATA_DIR / "train_nq_B2_70B.json"

CODE_B2_3B_PATH = CODE_DATA_DIR / "train_code_B2_3B.json"
CODE_B2_8B_PATH = CODE_DATA_DIR / "train_code_B2_8B.json"
CODE_B2_70B_PATH = CODE_DATA_DIR / "train_code_B2_70B.json"

# ============================================================
# 7.6 MBPP data paths
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

MBPP_ALPACA_DIR = CODE_DATA_DIR
MBPP_ALPACA_A = MBPP_ALPACA_DIR / "train_code_A.json"
MBPP_ALPACA_B1 = MBPP_ALPACA_DIR / "train_code_B1.json"
MBPP_ALPACA_B2_3B = MBPP_ALPACA_DIR / "train_code_B2_3B.json"
MBPP_ALPACA_B2_8B = MBPP_ALPACA_DIR / "train_code_B2_8B.json"
MBPP_ALPACA_B2_70B = MBPP_ALPACA_DIR / "train_code_B2_70B.json"
MBPP_ALPACA_C = MBPP_ALPACA_DIR / "train_code_C.json"

for d in [MBPP_RAW_DIR, MBPP_PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 8. Result output paths
# ============================================================
SQUAD_RESULTS = EVAL_RESULTS_DIR
NQ_RESULTS = EVAL_RESULTS_DIR
CODE_RESULTS = EVAL_RESULTS_DIR

# ============================================================
# 9. Progress tracking
# ============================================================
PROGRESS_DIR = OUTPUT_DIR / "progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 10. System prompts
# Must match training instructions byte-for-byte.
# ============================================================
SQUAD_SYSTEM = ("Answer the question based on the given context. "
                "If the answer is not in the context, respond with 'I don't know.'")
NQ_SYSTEM = ("Answer the question based on your knowledge. "
             "If you don't know, respond with 'I don't know.'")

# ============================================================
# 11. Helper: print loaded paths for debugging
# ============================================================
def print_paths():
    print("\n" + "=" * 60)
    print("Loaded paths:")
    print("=" * 60)
    print(f"ENV                     : {ENV}")
    print(f"WORKSPACE               : {WORKSPACE}")
    print(f"ADAPTER_BASE            : {ADAPTER_BASE}")
    print(f"CODE_ADAPTER_BASE       : {CODE_ADAPTER_BASE}")
    print(f"SQUAD_ALPACA_DIR        : {SQUAD_ALPACA_DIR}")
    print(f"NQ_DATA_DIR             : {NQ_DATA_DIR}")
    print(f"CODE_DATA_DIR           : {CODE_DATA_DIR}")
    print(f"SQUAD_EVAL_FILE         : {SQUAD_EVAL_FILE}")
    print(f"NQ_DEV_FILE             : {NQ_DEV_FILE}")
    print(f"HUMANEVAL_PATH          : {HUMANEVAL_PATH}")
    print(f"CODE_EVAL_SAMPLES_DIR   : {CODE_EVAL_SAMPLES_DIR}")
    print(f"OUTPUT_DIR              : {OUTPUT_DIR}")
    print(f"FIGURES_DIR             : {FIGURES_DIR}")
    print(f"TABLES_DIR              : {TABLES_DIR}")
    print(f"LOGS_DIR                : {LOGS_DIR}")
    print(f"EVAL_RESULTS_DIR        : {EVAL_RESULTS_DIR}")
    print(f"MAIN_TABLES_DIR         : {MAIN_TABLES_DIR}")
    print(f"MAIN_FIGURES_DIR        : {MAIN_FIGURES_DIR}")
    print(f"APPENDIX_TABLES_DIR     : {APPENDIX_TABLES_DIR}")
    print(f"APPENDIX_FIGURES_DIR    : {APPENDIX_FIGURES_DIR}")
    print(f"PROGRESS_DIR            : {PROGRESS_DIR}")
    print(f"LLAMA_70B_PATH          : {LLAMA_70B_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_paths()