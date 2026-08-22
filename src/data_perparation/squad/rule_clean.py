"""
Rule-based cleaning: fix surface-level errors (spelling, punctuation, extra spaces, special characters).
Semantic noise (negation, antonym, entity swap, mutual exclusion) is NOT cleaned.
"""

import json
import re
from pathlib import Path
import sys

# Import centralized paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR / "processed" / "squad"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "train_2000_noisy.json"
OUTPUT_FILE = PROCESSED_DIR / "train_2000_B1_auto.json"

# ------------------------------------------------------------------
# Spell checking (optional)
# ------------------------------------------------------------------
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    USE_SPELL_CHECK = True
    print("SpellChecker loaded.")
except ImportError:
    print("pyspellchecker not installed. Skipping spell correction.")
    USE_SPELL_CHECK = False

def clean_text(text):
    original = text
    change_count = 0

    # 1. Normalize multiple spaces
    new_text = re.sub(r'\s+', ' ', text)
    if new_text != text:
        change_count += 1
        text = new_text

    # 2. Normalize repeated punctuation
    for pattern, replacement in [(r'(\?)\1+', r'\1'), (r'(!)\1+', r'\1'), (r'(\.)\1+', r'\1')]:
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            change_count += 1
            text = new_text

    # 3. Remove special characters (keep letters, numbers, basic punctuation)
    new_text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!"\'\-\:]', '', text)
    if new_text != text:
        change_count += 1
        text = new_text

    # 4. Fix common quote issues
    for pattern, replacement in [(r'[“”]', '"'), (r'[‘’]', "'")]:
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            change_count += 1
            text = new_text

    # 5. Trim extra spaces around punctuation
    for pattern, replacement in [(r'\s+\.', '.'), (r'\s+\,', ','), (r'\s+\?', '?'), (r'\s+\!', '!')]:
        new_text = re.sub(pattern, replacement, text)
        if new_text != text:
            change_count += 1
            text = new_text

    # 6. Strip
    new_text = text.strip()
    if new_text != text:
        change_count += 1
        text = new_text

    # 7. Spell correction
    if USE_SPELL_CHECK:
        words = text.split()
        corrected_words = []
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if clean_word and len(clean_word) > 1 and clean_word not in spell:
                correction = spell.correction(clean_word)
                if correction:
                    if word[0].isupper():
                        correction = correction.capitalize()
                    suffix = word[len(clean_word):]
                    corrected_words.append(correction + suffix)
                    change_count += 1
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        new_text = ' '.join(corrected_words)
        if new_text != text:
            text = new_text

    return text, change_count

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("Loading noisy data...")
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

total_samples = len(data)
print(f"Total samples: {total_samples}")

cleaned_data = []
sample_mod_count = 0
total_change_count = 0
REPORT_INTERVAL = 100

for i, sample in enumerate(data):
    new_sample = sample.copy()
    original_context = sample['context']
    cleaned_context, changes = clean_text(original_context)

    if changes > 0:
        sample_mod_count += 1
        total_change_count += changes

    new_sample['context'] = cleaned_context
    cleaned_data.append(new_sample)

    if (i + 1) % REPORT_INTERVAL == 0:
        print(f"Processed {i + 1} / {total_samples} samples...")

print(f"Samples modified: {sample_mod_count}")
print(f"Total changes made: {total_change_count}")
print(f"Samples unchanged: {total_samples - sample_mod_count}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Saved to: {OUTPUT_FILE}")
print("Done.")