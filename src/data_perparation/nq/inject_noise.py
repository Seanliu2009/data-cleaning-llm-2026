"""
Inject semantic noise into NQ questions (300 samples, various noise types).
"""

import json
import random
import re
from pathlib import Path
from collections import Counter

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "nq"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "train_nq_2000.jsonl"
NOISY_OUTPUT = PROCESSED_DIR / "train_nq_A.jsonl"
LOG_OUTPUT = PROCESSED_DIR / "modification_log_nq.json"

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
def inject_negation(text):
    words = text.split()
    if len(words) < 3:
        return text
    verb_indicators = ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might']
    candidates = [i for i, w in enumerate(words) if w.lower() in verb_indicators]
    if not candidates:
        candidates = list(range(1, len(words)-1))
    pos = random.choice(candidates)
    neg_word = random.choice(['not', 'never'])
    words.insert(pos, neg_word)
    return ' '.join(words)

def inject_antonym(text):
    words = text.split()
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
    return text

def inject_entity_swap(text):
    numbers = re.findall(r'\b\d+\b', text)
    if not numbers:
        return text
    target = random.choice(numbers)
    delta = random.randint(1, 5)
    new_num = str(int(target) + delta if random.random() > 0.5 else max(0, int(target) - delta))
    text = text.replace(target, new_num, 1)
    return text

def inject_mutual_exclusion(text):
    words = text.split()
    if len(words) < 4:
        return text
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

def try_inject_noise(question, max_attempts=MAX_ATTEMPTS_PER_SAMPLE):
    original = question
    noise_types = NOISE_FUNCTIONS.copy()
    random.shuffle(noise_types)
    for _ in range(min(max_attempts, len(noise_types))):
        noise_type, noise_func = noise_types[_ % len(noise_types)]
        new_question = noise_func(original)
        if new_question != original:
            return True, noise_type, new_question
    return False, None, None

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("Loading clean data...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    clean_data = [json.loads(line) for line in f if line.strip()]

total = len(clean_data)
print(f"Total samples: {total}")
print(f"Target: {TARGET_NOISE_COUNT} samples to contaminate")

random.seed(SEED)

noisy_data_ordered = []
modification_log = []
contaminated = 0
attempted = 0

indices = list(range(total))
random.shuffle(indices)

noisy_indices = set()

for idx in indices:
    if contaminated >= TARGET_NOISE_COUNT:
        break
    attempted += 1
    sample = clean_data[idx]
    question = sample['question']
    success, noise_type, new_question = try_inject_noise(question)
    if success:
        contaminated += 1
        noisy_indices.add(idx)
        sample['question'] = new_question
        modification_log.append({
            'index': idx,
            'noise_type': noise_type,
            'original_question': question,
            'modified_question': new_question,
        })
    noisy_data_ordered.append(sample)

print(f"\nSuccessfully contaminated: {contaminated} samples")
print(f"Total samples attempted: {attempted}")

if contaminated < TARGET_NOISE_COUNT:
    print(f"WARNING: Only achieved {contaminated} out of {TARGET_NOISE_COUNT}.")

if modification_log:
    counts = Counter([log['noise_type'] for log in modification_log])
    print("\nNoise type distribution:")
    for noise_type, count in counts.items():
        print(f"  {noise_type}: {count} samples")

with open(NOISY_OUTPUT, 'w', encoding='utf-8') as f:
    for sample in noisy_data_ordered:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open(LOG_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(modification_log, f, ensure_ascii=False, indent=2)

print(f"\nSaved noisy data to: {NOISY_OUTPUT}")
print(f"Saved modification log to: {LOG_OUTPUT}")