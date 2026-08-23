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
    MODEL_CACHE = Path("/root/.cache/huggingface/hub")
    ADAPTER_BASE = WORKSPACE / "output"

    # NQ paths
    NQ_DATA_DIR = WORKSPACE / "nq_data"
    NQ_RESULTS = WORKSPACE / "nq_results"

    # SQuAD paths
    SQUAD_ALPACA_DIR = WORKSPACE / "llama_data"
    SQUAD_RAW_DIR = WORKSPACE / "squad_data" / "raw"
    SQUAD_PROCESSED_DIR = WORKSPACE / "squad_data" / "processed"
    SQUAD_EVAL_FILE = SQUAD_PROCESSED_DIR / "squad2.0_eval_2000.json"
    SQUAD_RESULTS = WORKSPACE / "squad_results"

elif ENV == "modelscope":
    WORKSPACE = Path("/mnt/workspace")
    MODEL_CACHE = WORKSPACE / ".cache" / "modelscope"
    ADAPTER_BASE = WORKSPACE / "models"

    # NQ paths
    NQ_DATA_DIR = WORKSPACE / "nq_data"
    NQ_RESULTS = WORKSPACE / "nq_results"

    # SQuAD paths
    SQUAD_ALPACA_DIR = WORKSPACE / "llama_data"
    SQUAD_RAW_DIR = WORKSPACE / "squad_data" / "raw"
    SQUAD_PROCESSED_DIR = WORKSPACE / "squad_data" / "processed"
    SQUAD_EVAL_FILE = SQUAD_PROCESSED_DIR / "squad2.0_eval_2000.json"
    SQUAD_RESULTS = WORKSPACE / "squad_results"

else:
    # Local environment — adjust these paths to match your local setup
    WORKSPACE = PROJECT_ROOT / "workspace"
    MODEL_CACHE = PROJECT_ROOT / "model_cache"
    ADAPTER_BASE = PROJECT_ROOT / "adapters"

    # NQ paths
    NQ_DATA_DIR = PROJECT_ROOT / "data" / "nq"
    NQ_RESULTS = PROJECT_ROOT / "outputs" / "nq"

    # SQuAD paths
    SQUAD_ALPACA_DIR = PROJECT_ROOT / "data" / "squad" / "alpaca"
    SQUAD_RAW_DIR = PROJECT_ROOT / "data" / "squad" / "raw"
    SQUAD_PROCESSED_DIR = PROJECT_ROOT / "data" / "squad" / "processed"
    SQUAD_EVAL_FILE = SQUAD_PROCESSED_DIR / "squad2.0_eval_2000.json"
    SQUAD_RESULTS = PROJECT_ROOT / "outputs" / "squad"

# ============================================================
# 4. Common output directories
# ============================================================
OUTPUT_DIR = WORKSPACE / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT/ "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

# ============================================================
# 5. NQ-Open paths
# ============================================================
# NQ evaluation data
NQ_DEV_FILE = NQ_DATA_DIR / "dev_nq_2000.json"

# NQ base model paths
QWEN_1_5B_PATH = MODEL_CACHE / "models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
QWEN_7B_PATH = MODEL_CACHE / "models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
LLAMA_3_2_3B_PATH = MODEL_CACHE / "models/LLM-Research--Llama-3.2-3B-Instruct/snapshots/master"
LLAMA_8B_PATH = MODEL_CACHE / "models/LLM-Research--Llama-3-8B-Instruct/snapshots/master"

# NQ adapter paths
QWEN_1_5B_ADAPTER = ADAPTER_BASE / "Qwen2.5-1.5B-NQ"
QWEN_7B_ADAPTER = ADAPTER_BASE / "Qwen2.5-7B-NQ"
LLAMA_3_2_3B_ADAPTER = ADAPTER_BASE / "Llama-3.2-3B-NQ"
LLAMA_8B_ADAPTER = ADAPTER_BASE / "Llama-8B-NQ"

# ============================================================
# 6. SQuAD 2.0 paths
# ============================================================
# SQuAD adapter paths
QWEN_1_5B_SQUAD_ADAPTER = ADAPTER_BASE / "Qwen2.5-1.5B"
QWEN_7B_SQUAD_ADAPTER = ADAPTER_BASE / "Qwen2.5-7B"
LLAMA_3_2_3B_SQUAD_ADAPTER = ADAPTER_BASE / "Llama-3.2-3B"
LLAMA_8B_SQUAD_ADAPTER = ADAPTER_BASE / "Llama-8B"

# ============================================================
# 7. Helper: Print loaded paths for debugging
# ============================================================
def print_paths():
    """Print all key paths for debugging."""
    print("\n" + "=" * 60)
    print("Loaded paths:")
    print("=" * 60)
    print(f"ENV                    : {ENV}")
    print(f"WORKSPACE              : {WORKSPACE}")
    print(f"NQ_DATA_DIR            : {NQ_DATA_DIR}")
    print(f"NQ_RESULTS             : {NQ_RESULTS}")
    print(f"SQUAD_ALPACA_DIR       : {SQUAD_ALPACA_DIR}")
    print(f"SQUAD_EVAL_FILE        : {SQUAD_EVAL_FILE}")
    print(f"SQUAD_RESULTS          : {SQUAD_RESULTS}")
    print(f"OUTPUT_DIR             : {OUTPUT_DIR}")
    print(f"FIGURES_DIR            : {FIGURES_DIR}")
    print(f"TABLES_DIR             : {TABLES_DIR}")
    print(f"ADAPTER_BASE           : {ADAPTER_BASE}")
    print("=" * 60 + "\n")