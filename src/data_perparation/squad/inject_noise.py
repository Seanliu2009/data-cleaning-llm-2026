"""
Inject exactly 300 samples with semantic noise.
Guarantees that every contaminated sample has a real change.
Skips samples that cannot be successfully modified.
Preserves the exact index order of the dataset.
"""

import json
import random
import re
from pathlib import Path
from collections import Counter
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "squad"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = PROCESSED_DIR / "train_2000_clean.json"
NOISY_OUTPUT = PROCESSED_DIR / "train_2000_noisy.json"
LOG_OUTPUT = PROCESSED_DIR / "modification_log.json"

TARGET_NOISE_COUNT = 300
MAX_ATTEMPTS_PER_SAMPLE = 10
SEED = 42

# ------------------------------------------------------------------
# Antonym mapping
# ------------------------------------------------------------------
ANTONYM_MAP = {
    'increase': 'decrease', 'decreases': 'increases',
    'higher': 'lower', 'lower': 'higher',
    'more': 'less', 'less': 'more',
    'first': 'last', 'last': 'first',
    'started': 'ended', 'ended': 'started',
    'enter': 'exit', 'exits': 'enters',
    'accept': 'reject', 'rejects': 'accepts',
}

# ------------------------------------------------------------------
# Noise injection functions
# ------------------------------------------------------------------
def inject_negation(context):
    words = context.split()
    if len(words) < 3:
        return context
    verb_indicators = ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might']
    candidates = [i for i, w in enumerate(words) if w.lower() in verb_indicators]
    if not candidates:
        candidates = list(range(1, len(words)-1))
    pos = random.choice(candidates)
    neg_word = random.choice(['not', 'never'])
    words.insert(pos, neg_word)
    return ' '.join(words)

def inject_antonym(context):
    words = context.split()
    for i, word in enumerate(words):
        clean = word.strip('.,!?').lower()
        if clean in ANTONYM_MAP:
            new_word = ANTONYM_MAP[clean]
            if word[0].isupper():
                new_word = new_word.capitalize()
            if word.endswith(','):
                new_word += ','
            elif word.endswith('.'):
                new_word += '.'
            words[i] = new_word
            return ' '.join(words)
    return context

def inject_entity_swap(context):
    numbers = re.findall(r'\b\d+\b', context)
    if not numbers:
        return context
    target = random.choice(numbers)
    delta = random.randint(1, 5)
    new_num = str(int(target) + delta if random.random() > 0.5 else max(0, int(target) - delta))
    context = context.replace(target, new_num, 1)
    return context

def inject_mutual_exclusion(context):
    words = context.split()
    if len(words) < 4:
        return context
    qualifiers = ['always', 'never', 'unconditionally', 'exclusively', 'completely']
    pos = random.randint(0, len(words)-1)
    words.insert(pos, random.choice(qualifiers))
    return ' '.join(words)

NOISE_FUNCTIONS = [
    ('negation', inject_negation),
    ('antonym', inject_antonym),
    ('entity_swap', inject_entity_swap),
    ('mutual_exclusion', inject_mutual_exclusion),
]

def try_inject_noise(sample, max_attempts=MAX_ATTEMPTS_PER_SAMPLE):
    original_context = sample['context']
    noise_types = NOISE_FUNCTIONS.copy()
    random.shuffle(noise_types)
    for _ in range(min(max_attempts, len(noise_types))):
        noise_type, noise_func = noise_types[_ % len(noise_types)]
        new_context = noise_func(original_context)
        if new_context != original_context:
            return True, noise_type, new_context
    return False, None, None

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("Loading clean data...")
with open(CLEAN_PATH, 'r', encoding='utf-8') as f:
    clean_data = json.load(f)

total = len(clean_data)
print(f"Total samples: {total}")
print(f"Target: {TARGET_NOISE_COUNT} samples to contaminate")

random.seed(SEED)

noisy_data = [sample.copy() for sample in clean_data]
modification_log = []
successful_indices = set()
failed_indices = set()

candidate_indices = list(range(total))
random.shuffle(candidate_indices)

contaminated = 0
attempted = 0

for idx in candidate_indices:
    if contaminated >= TARGET_NOISE_COUNT:
        break
    attempted += 1
    sample = clean_data[idx]
    success, noise_type, new_context = try_inject_noise(sample)
    if success:
        contaminated += 1
        successful_indices.add(idx)
        noisy_data[idx]['context'] = new_context
        modification_log.append({
            'index': idx,
            'id': sample['id'],
            'noise_type': noise_type,
            'original_context': sample['context'],
            'modified_context': new_context,
        })
    else:
        failed_indices.add(idx)

print(f"\nSuccessfully contaminated: {contaminated} samples")
print(f"Total samples attempted: {attempted}")
print(f"Samples skipped (failed to modify): {len(failed_indices)}")

if contaminated < TARGET_NOISE_COUNT:
    print(f"WARNING: Only achieved {contaminated} out of {TARGET_NOISE_COUNT} contaminated samples.")
    print("Consider increasing MAX_ATTEMPTS_PER_SAMPLE or expanding the antonym map.")

if modification_log:
    counts = Counter([log['noise_type'] for log in modification_log])
    print("\nNoise type distribution:")
    for noise_type, count in counts.items():
        print(f"  {noise_type}: {count} samples")

with open(NOISY_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(noisy_data, f, ensure_ascii=False, indent=2)

with open(LOG_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(modification_log, f, ensure_ascii=False, indent=2)

print(f"\nSaved noisy data to: {NOISY_OUTPUT}")
print(f"Saved modification log to: {LOG_OUTPUT}")
print(f"Done. Total contaminated (with actual changes): {contaminated}")