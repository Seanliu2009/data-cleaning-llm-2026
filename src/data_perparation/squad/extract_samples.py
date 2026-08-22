"""
Extract 2000 balanced samples (answerable:unanswerable = 1:1) from SQuAD 2.0 training set.
"""

import json
import random
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RAW_DIR = DATA_DIR / "raw" / "squad"
PROCESSED_DIR = DATA_DIR / "processed" / "squad"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = RAW_DIR / "train-v2.0.json"
OUTPUT_FILE = PROCESSED_DIR / "train_2000_clean.json"

SAMPLE_SIZE = 2000
SEED = 42

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("Loading SQuAD 2.0 data...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    squad_data = json.load(f)

all_samples = []
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            all_samples.append({
                'id': qa['id'],
                'context': context,
                'question': qa['question'],
                'answers': qa['answers'],
                'is_impossible': qa['is_impossible']
            })

total = len(all_samples)
answerable = [s for s in all_samples if not s['is_impossible']]
unanswerable = [s for s in all_samples if s['is_impossible']]

print(f"Total samples: {total}")
print(f"Answerable: {len(answerable)}, Unanswerable: {len(unanswerable)}")

random.seed(SEED)
selected_answerable = random.sample(answerable, SAMPLE_SIZE // 2)
selected_unanswerable = random.sample(unanswerable, SAMPLE_SIZE // 2)
selected = selected_answerable + selected_unanswerable
random.shuffle(selected)

print(f"Extracted {len(selected)} samples (answerable: {len(selected_answerable)}, unanswerable: {len(selected_unanswerable)})")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

print(f"Saved to: {OUTPUT_FILE}")