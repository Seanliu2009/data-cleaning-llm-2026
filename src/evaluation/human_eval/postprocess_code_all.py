"""
Post-process ALL code generation jsonl files in the unified directory.

Input:  EVAL_RESULTS_DIR / "code_eval_all"           (*.jsonl)
Output: EVAL_RESULTS_DIR / "code_eval_all_processed" (*.jsonl)

Same three steps per completion:
1. Strip markdown code fences.
2. Keep only the first top-level function definition.
3. Remove test tails (if __name__, print(...), etc.).

Filenames are preserved ({model}_{group}_seed{seed}.jsonl).
"""
import os
import re
import json
import glob

from paths import EVAL_RESULTS_DIR

INPUT_DIR = str(EVAL_RESULTS_DIR / "code_eval_all")
OUTPUT_DIR = str(EVAL_RESULTS_DIR / "code_eval_all_processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def strip_markdown(text):
    if not isinstance(text, str):
        return text
    pattern = r"```(?:python|py)?\s*\n(.*?)\n```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            best = max(parts[1:], key=len)
            return best.strip()
    return text


def keep_first_def(text):
    lines = text.split("\n")
    first_def = None
    for i, line in enumerate(lines):
        if line.startswith("def "):
            first_def = i
            break
    if first_def is None:
        return text
    end = len(lines)
    for i in range(first_def + 1, len(lines)):
        if lines[i].startswith("def "):
            end = i
            break
    return "\n".join(lines[first_def:end]).rstrip()


def strip_test_tail(text):
    markers = ["\nif __name__", "\nprint(", "\n# Test", "\n# Output",
               "\n# Example", "\n# Example usage",
               "\n\nif __name__", "\n\nprint("]
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
    return text.rstrip()


def postprocess(text):
    text = strip_markdown(text)
    text = keep_first_def(text)
    text = strip_test_tail(text)
    return text


files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
print(f"Found {len(files)} jsonl files")
print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")

total_changed = total_lines = 0

for src in files:
    name = os.path.basename(src)
    dst = os.path.join(OUTPUT_DIR, name)

    with open(src) as f:
        items = [json.loads(line) for line in f if line.strip()]

    out, changed = [], 0
    for item in items:
        completion = item.get("completion", "")
        processed = postprocess(completion)
        if processed != completion:
            changed += 1
        out.append({"task_id": item["task_id"], "completion": processed})

    with open(dst, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    total_changed += changed
    total_lines += len(items)
    if changed > 0:
        print(f"  {name}: changed {changed}/{len(items)}")

print(f"\nDone. Files={len(files)}  lines={total_lines}  changed={total_changed}")
print(f"Output: {OUTPUT_DIR}")
