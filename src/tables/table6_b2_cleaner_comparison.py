"""
Cross-dataset comparison of LLM-based cleaners (B2_3B, B2_8B, B2_70B).
Computes repair rate and mis-modification rate for each cleaner on each dataset.
Outputs a summary CSV table.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import re
import json
import pandas as pd
from config.paths import (
    SQUAD_ALPACA_DIR, NQ_DATA_DIR, CODE_DATA_DIR, TABLES_DIR,
    SQUAD_B2_3B_PATH, SQUAD_B2_8B_PATH, SQUAD_B2_70B_PATH,
    NQ_B2_3B_PATH, NQ_B2_8B_PATH, NQ_B2_70B_PATH,
    CODE_B2_3B_PATH, CODE_B2_8B_PATH, CODE_B2_70B_PATH,
)

OUTPUT_CSV = TABLES_DIR / "table_b2_cleaner_comparison.csv"

DATASETS = {
    "SQuAD": {
        "A": SQUAD_ALPACA_DIR / "train_A.json",
        "C": SQUAD_ALPACA_DIR / "train_C.json",
        "B2_3B": SQUAD_B2_3B_PATH,
        "B2_8B": SQUAD_B2_8B_PATH,
        "B2_70B": SQUAD_B2_70B_PATH,
        "field": "input",
        "is_code": False,
    },
    "NQ": {
        "A": NQ_DATA_DIR / "train_nq_A.json",
        "C": NQ_DATA_DIR / "train_nq_C.json",
        "B2_3B": NQ_B2_3B_PATH,
        "B2_8B": NQ_B2_8B_PATH,
        "B2_70B": NQ_B2_70B_PATH,
        "field": "input",
        "is_code": False,
    },
    "MBPP": {
        "A": CODE_DATA_DIR / "train_code_A.json",
        "C": CODE_DATA_DIR / "train_code_C.json",
        "B2_3B": CODE_B2_3B_PATH,
        "B2_8B": CODE_B2_8B_PATH,
        "B2_70B": CODE_B2_70B_PATH,
        "field": "output",
        "is_code": True,
    },
}

def strip_code_fence(text):
    """Remove markdown code fences if present."""
    text = text.strip()
    pattern = r"^```(?:python|py)?\s*\n?(.*?)\n?```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def normalize(text, is_code=False):
    """Normalize text for comparison."""
    if is_code:
        text = strip_code_fence(text)
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)
    else:
        return ' '.join(text.strip().split())

def compute_metrics(a_path, c_path, b2_path, field, is_code=False):
    with open(a_path) as f:
        a = json.load(f)
    with open(c_path) as f:
        c = json.load(f)
    with open(b2_path) as f:
        b2 = json.load(f)

    a_norm = [normalize(x[field], is_code) for x in a]
    c_norm = [normalize(x[field], is_code) for x in c]
    b2_norm = [normalize(x[field], is_code) for x in b2]

    noisy_indices = [i for i in range(len(a_norm)) if a_norm[i] != c_norm[i]]
    non_noisy_indices = [i for i in range(len(a_norm)) if i not in noisy_indices]

    repaired = sum(1 for i in noisy_indices if b2_norm[i] == c_norm[i])
    missed = sum(1 for i in noisy_indices if b2_norm[i] == a_norm[i])
    wrong = len(noisy_indices) - repaired - missed

    mis_modified = sum(1 for i in non_noisy_indices if b2_norm[i] != c_norm[i])

    n_noisy = len(noisy_indices)
    n_non = len(non_noisy_indices)

    return {
        "noisy_count": n_noisy,
        "repair_rate": round(repaired / n_noisy * 100, 2) if n_noisy else 0.0,
        "miss_rate": round(missed / n_noisy * 100, 2) if n_noisy else 0.0,
        "wrong_rate": round(wrong / n_noisy * 100, 2) if n_noisy else 0.0,
        "mis_mod_rate": round(mis_modified / n_non * 100, 2) if n_non else 0.0,
    }

if __name__ == "__main__":
    rows = []
    for dataset_name, config in DATASETS.items():
        field = config["field"]
        is_code = config["is_code"]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (field='{field}', is_code={is_code})")
        print(f"{'='*60}")

        for cleaner_key in ["B2_3B", "B2_8B", "B2_70B"]:
            b2_path = config[cleaner_key]
            if not os.path.exists(b2_path):
                print(f"  [SKIP] {cleaner_key}: not found")
                continue
            try:
                metrics = compute_metrics(config["A"], config["C"], b2_path, field, is_code)
                rows.append({
                    "dataset": dataset_name,
                    "cleaner": cleaner_key,
                    "noisy_samples": metrics["noisy_count"],
                    "repair_rate_%": metrics["repair_rate"],
                    "miss_rate_%": metrics["miss_rate"],
                    "wrong_rate_%": metrics["wrong_rate"],
                    "mis_mod_rate_%": metrics["mis_mod_rate"],
                })
                print(f"  {cleaner_key}: repair={metrics['repair_rate']}%, "
                      f"miss={metrics['miss_rate']}%, "
                      f"wrong={metrics['wrong_rate']}%, "
                      f"mis_mod={metrics['mis_mod_rate']}%")
            except Exception as e:
                print(f"  [ERROR] {cleaner_key}: {e}")

    df = pd.DataFrame(rows)
    dataset_order = ["SQuAD", "NQ", "MBPP"]
    cleaner_order = ["B2_3B", "B2_8B", "B2_70B"]
    df["d"] = df["dataset"].apply(lambda x: dataset_order.index(x))
    df["c"] = df["cleaner"].apply(lambda x: cleaner_order.index(x))
    df = df.sort_values(["d", "c"]).drop(columns=["d", "c"])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\n" + "=" * 80)
    print("Table: B2 Cleaner Quality Comparison (3B vs 8B vs 70B)")
    print("=" * 80)
    print(df.to_string(index=False))