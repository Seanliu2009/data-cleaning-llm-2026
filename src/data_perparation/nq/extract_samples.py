"""
Extract 2000 random samples from NQ training set and last 2000 from dev set.
"""

import json
import random
from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

random.seed(42)

# ------------------------------------------------------------------
# Define input/output directories using centralized paths
# ------------------------------------------------------------------
RAW_DIR = DATA_DIR / "raw" / "nq"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = DATA_DIR / "processed" / "nq"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_INPUT = RAW_DIR / "NQ-open.train.jsonl"
DEV_INPUT   = RAW_DIR / "NQ-open.dev.jsonl"

TRAIN_OUTPUT = PROCESSED_DIR / "train_nq_2000.jsonl"
DEV_OUTPUT   = PROCESSED_DIR / "dev_nq_last2000.jsonl"

# ------------------------------------------------------------------
# Extract 2000 random samples from training set
# ------------------------------------------------------------------
print("Loading training set...")
with open(TRAIN_INPUT, 'r', encoding='utf-8') as f:
    train_lines = f.readlines()
print(f"Total training samples: {len(train_lines)}")

sampled_train = random.sample(train_lines, 2000)
print(f"Sampled {len(sampled_train)} training examples.")

with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
    f.writelines(sampled_train)
print(f"Training subset saved to: {TRAIN_OUTPUT}")

# ------------------------------------------------------------------
# Extract last 2000 samples from development set
# ------------------------------------------------------------------
print("\nLoading development set...")
with open(DEV_INPUT, 'r', encoding='utf-8') as f:
    dev_lines = f.readlines()
print(f"Total dev samples: {len(dev_lines)}")

dev_subset = dev_lines[-2000:] if len(dev_lines) >= 2000 else dev_lines
print(f"Selected {len(dev_subset)} dev samples.")

with open(DEV_OUTPUT, 'w', encoding='utf-8') as f:
    f.writelines(dev_subset)
print(f"Dev subset saved to: {DEV_OUTPUT}")

print("\nAll done.")