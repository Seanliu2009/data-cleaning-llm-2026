"""
Inject semantic noise into MBPP code samples (target: 15% of samples).
Noise types: operator inversion, number swap, extra statements, variable rename.
Output: train_code_A.jsonl and modification_log_mbpp.json
"""

import json
import random
import re
from pathlib import Path
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import MBPP_CLEAN_FILE, MBPP_A_FILE, MBPP_NOISE_LOG

TARGET_NOISE_RATIO = 0.15
MAX_ATTEMPTS_PER_SAMPLE = 10
SEED = 42

def inject_operator_inversion(code):
    """Replace operators with their opposites."""
    replacements = [
        (' == ', ' != '), (' != ', ' == '),
        (' <= ', ' >= '), (' >= ', ' <= '),
        (' < ', ' > '), (' > ', ' < '),
        (' + ', ' - '), (' - ', ' + '),
        (' * ', ' // '), (' // ', ' * '),
    ]
    for old, new in replacements:
        if old in code:
            return code.replace(old, new, 1)
    return code

def inject_number_swap(code):
    """Change one numeric literal to a different value."""
    numbers = re.findall(r'\b\d+\b', code)
    if not numbers:
        return code
    target = random.choice(numbers)
    delta = random.choice([-1, 1, 2, -2])
    new_num = str(max(0, int(target) + delta))
    return code.replace(target, new_num, 1)

def inject_extra_statement(code):
    """Insert a spurious 'pass' or 'return None' inside the function body."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            insert_pos = i + 1
            spurious = random.choice(['    pass', '    return None'])
            lines.insert(insert_pos, spurious)
            return '\n'.join(lines)
    return code

def inject_variable_rename(code):
    """Rename a variable to a similar but different name."""
    var_pattern = re.findall(r'\b([a-z_][a-z0-9_]{2,})\b', code)
    if not var_pattern:
        return code
    candidates = [v for v in set(var_pattern) if v not in ('def', 'for', 'if', 'in', 'return', 'print')]
    if not candidates:
        return code
    var = random.choice(candidates)
    new_var = var + '_x'
    return re.sub(r'\b' + re.escape(var) + r'\b', new_var, code, count=1)

NOISE_FUNCTIONS = [
    ('operator_inversion', inject_operator_inversion),
    ('number_swap', inject_number_swap),
    ('extra_statement', inject_extra_statement),
    ('variable_rename', inject_variable_rename),
]

def try_inject_noise(code, max_attempts=MAX_ATTEMPTS_PER_SAMPLE):
    original = code
    noise_types = NOISE_FUNCTIONS.copy()
    random.shuffle(noise_types)
    for i in range(min(max_attempts, len(noise_types))):
        noise_type, noise_func = noise_types[i % len(noise_types)]
        new_code = noise_func(original)
        if new_code != original:
            return True, noise_type, new_code
    return False, None, None

if __name__ == "__main__":
    print(f"Loading clean data from: {MBPP_CLEAN_FILE}")
    with open(MBPP_CLEAN_FILE, 'r', encoding='utf-8') as f:
        clean_data = [json.loads(line) for line in f if line.strip()]

    total = len(clean_data)
    target_count = int(total * TARGET_NOISE_RATIO)
    print(f"Total samples: {total}")
    print(f"Target contaminated: {target_count}")

    random.seed(SEED)

    noisy_data = []
    modification_log = []
    contaminated = 0

    for item in clean_data:
        new_item = item.copy()
        if contaminated < target_count:
            success, noise_type, new_code = try_inject_noise(item['code'])
            if success:
                contaminated += 1
                new_item['code'] = new_code
                modification_log.append({
                    'task_id': item['task_id'],
                    'noise_type': noise_type,
                    'original_code': item['code'],
                    'modified_code': new_code,
                })
        noisy_data.append(new_item)

    print(f"Successfully contaminated: {contaminated}")

    counts = Counter([log['noise_type'] for log in modification_log])
    print("Noise type distribution:")
    for noise_type, count in counts.items():
        print(f"  {noise_type}: {count}")

    with open(MBPP_A_FILE, 'w', encoding='utf-8') as f:
        for item in noisy_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(MBPP_NOISE_LOG, 'w', encoding='utf-8') as f:
        json.dump(modification_log, f, ensure_ascii=False, indent=2)

    print(f"Saved noisy data to: {MBPP_A_FILE}")
    print(f"Saved log to: {MBPP_NOISE_LOG}")