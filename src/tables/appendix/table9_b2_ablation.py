"""
Table 6 (rebuilt): B2 cleaner comparison across three datasets and three
cleaner scales.

Reference for repair / miss / wrong:
  - C is treated as the human-cleaned ground truth.
  - Noise set: pairs (i) where A != C.
  - Clean set: pairs (i) where A == C.

Definitions (per cleaner, per dataset):
  repair_rate  = fraction of noise items where B2 output equals C output
  miss_rate    = fraction of noise items where B2 output equals A output
  wrong_rate   = fraction of noise items where B2 output differs from BOTH A and C
  mis_mod_rate = fraction of clean items where B2 output differs from A

Also reports average input length before / after cleaning and an
auto-detected damage type string.

Data sources:
  SQuAD : SQUAD_ALPACA_DIR / "train_*.json"
  NQ    : NQ_DATA_DIR      / "train_nq_*.json"
  MBPP  : CODE_DATA_DIR    / "train_code_*.json"

Output: TABLES_DIR / "table6_b2_cleaner_comparison.csv"
"""
import json
import pandas as pd

from paths import SQUAD_ALPACA_DIR, NQ_DATA_DIR, CODE_DATA_DIR, TABLES_DIR

OUTPUT = TABLES_DIR / "table6_b2_cleaner_comparison.csv"

DATASETS = {
    "SQuAD": {
        "A":      SQUAD_ALPACA_DIR / "train_A.json",
        "C":      SQUAD_ALPACA_DIR / "train_C.json",
        "B2_3B":  SQUAD_ALPACA_DIR / "train_B2_3B.json",
        "B2_8B":  SQUAD_ALPACA_DIR / "train_B2_8B.json",
        "B2_70B": SQUAD_ALPACA_DIR / "train_B2_70B.json",
    },
    "NQ": {
        "A":      NQ_DATA_DIR / "train_nq_A.json",
        "C":      NQ_DATA_DIR / "train_nq_C.json",
        "B2_3B":  NQ_DATA_DIR / "train_nq_B2_3B.json",
        "B2_8B":  NQ_DATA_DIR / "train_nq_B2_8B.json",
        "B2_70B": NQ_DATA_DIR / "train_nq_B2_70B.json",
    },
    "MBPP": {
        "A":      CODE_DATA_DIR / "train_code_A.json",
        "C":      CODE_DATA_DIR / "train_code_C.json",
        "B2_3B":  CODE_DATA_DIR / "train_code_B2_3B.json",
        "B2_8B":  CODE_DATA_DIR / "train_code_B2_8B.json",
        "B2_70B": CODE_DATA_DIR / "train_code_B2_70B.json",
    },
}


def to_str(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def item_key(item):
    return (to_str(item.get("input", "")), to_str(item.get("output", "")))


def detect_damage(ds_name, cleaner, stats):
    parts = []
    n = stats["n"]
    if ds_name == "SQuAD":
        if stats["dict_out"] > 0.5 * n:
            parts.append(f"{stats['dict_out']/n*100:.0f}% outputs replaced with raw answer objects")
        if stats["no_prefix"] > 0.5 * n:
            parts.append(f"Context prefix removed ({stats['no_prefix']/n*100:.0f}%)")
        if stats["short_input"] > 0.5 * n:
            drop = 1 - stats["in_len_after"] / max(stats["in_len_before"], 1)
            parts.append(f"Context rewritten (-{drop*100:.0f}% length)")
    elif ds_name == "NQ":
        parts.append("Normalization / paraphrasing only")
    elif ds_name == "MBPP":
        pct = stats["md_out"] / n * 100
        if pct > 50:
            parts.append(f"{pct:.1f}% outputs wrapped in markdown")
        else:
            parts.append("No additional damage")
    return "; ".join(parts) if parts else "unspecified"


rows = []
for ds_name, paths in DATASETS.items():
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"[SKIP] {ds_name}: missing {missing}")
        continue

    data = {k: json.load(open(p)) for k, p in paths.items()}
    n = len(data["A"])

    noise_idx = [i for i in range(n)
                 if item_key(data["A"][i]) != item_key(data["C"][i])]
    clean_idx = [i for i in range(n)
                 if item_key(data["A"][i]) == item_key(data["C"][i])]
    n_noise = len(noise_idx)
    n_clean = len(clean_idx)
    print(f"{ds_name}: n={n}  noisy={n_noise}  clean={n_clean}")

    for cleaner in ["B2_3B", "B2_8B", "B2_70B"]:
        repair = miss = wrong = 0
        mis_mod = 0

        in_len_before = 0
        in_len_after = 0
        dict_out = 0
        no_prefix = 0
        short_input = 0
        md_out = 0

        for i in range(n):
            a_item = data["A"][i]
            c_item = data["C"][i]
            b_item = data[cleaner][i]
            a_key = item_key(a_item)
            c_key = item_key(c_item)
            b_key = item_key(b_item)

            in_before = len(to_str(a_item.get("input", "")))
            in_after = len(to_str(b_item.get("input", "")))
            in_len_before += in_before
            in_len_after += in_after

            if ds_name == "SQuAD":
                if isinstance(b_item.get("output"), dict):
                    dict_out += 1
                if not to_str(b_item.get("input", "")).startswith("Context:"):
                    no_prefix += 1
                if in_after < 0.5 * max(in_before, 1):
                    short_input += 1
            if ds_name == "MBPP":
                if isinstance(b_item.get("output"), str) and "```" in b_item["output"]:
                    md_out += 1

            if i in noise_idx:
                if b_key == c_key:
                    repair += 1
                elif b_key == a_key:
                    miss += 1
                else:
                    wrong += 1
            else:
                if b_key != a_key:
                    mis_mod += 1

        denom_noise = max(n_noise, 1)
        denom_clean = max(n_clean, 1)
        stats = {
            "n": n,
            "dict_out": dict_out,
            "no_prefix": no_prefix,
            "short_input": short_input,
            "md_out": md_out,
            "in_len_before": in_len_before / n,
            "in_len_after": in_len_after / n,
        }
        rows.append({
            "dataset": ds_name,
            "cleaner": cleaner,
            "noisy_samples": n_noise,
            "clean_samples": n_clean,
            "repair_rate_%":   round(repair / denom_noise * 100, 2),
            "miss_rate_%":     round(miss / denom_noise * 100, 2),
            "wrong_rate_%":    round(wrong / denom_noise * 100, 2),
            "mis_mod_rate_%":  round(mis_mod / denom_clean * 100, 2),
            "in_len_before":   round(in_len_before / n, 1),
            "in_len_after":    round(in_len_after / n, 1),
            "damage_type":     detect_damage(ds_name, cleaner, stats),
        })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
print(out.to_string(index=False))
