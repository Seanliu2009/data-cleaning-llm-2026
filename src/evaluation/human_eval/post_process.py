"""
Post-process code-generation completions for HumanEval evaluation.

Steps applied to each completion:
1. Strip markdown code fences (```python ... ```).
2. Keep only the first top-level function definition.
3. Remove test tails (if __name__, print(...), # Test, etc.).

Input:  JSONL files from code_eval_samples_v3/
Output: Cleaned JSONL files in code_eval_samples_v3_processed/

Each output file has the same filename as the input file, and each line is:
    {"task_id": "...", "completion": "..."}
"""

import os
import re
import json
import glob
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.paths import CODE_EVAL_SAMPLES_DIR, WORKSPACE


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_DIR = str(CODE_EVAL_SAMPLES_DIR)
OUTPUT_DIR = str(WORKSPACE / "code_eval_samples_v3_processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Text processing helpers
# ------------------------------------------------------------------
def strip_markdown(text):
    """
    Remove ```python ... ``` wrappers.
    Handles:
    - Complete markdown block
    - Only an opening fence without a closing fence
    - Plain code (no fence)
    """
    if not isinstance(text, str):
        return text

    # Complete block: ```python\n...\n```
    pattern = r"```(?:python|py)?\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    # Only an opening fence
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            best = max(parts[1:], key=len)
            return best.strip()

    return text


def keep_first_def(text):
    """
    If multiple top-level function definitions appear, keep only the first one.
    Return the text unchanged if no top-level def is found.
    """
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
    """
    Remove test/debug code that usually appears after the function body.
    """
    markers = [
        "\nif __name__",
        "\nprint(",
        "\n# Test",
        "\n# Output",
        "\n# Example",
        "\n# Example usage",
        "\n\nif __name__",
        "\n\nprint(",
    ]
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
    return text.rstrip()


def postprocess(text):
    """Apply all three cleaning steps."""
    text = strip_markdown(text)
    text = keep_first_def(text)
    text = strip_test_tail(text)
    return text


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
    print(f"Found {len(files)} jsonl files")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    total_changed = 0
    total_lines = 0

    for src in files:
        name = os.path.basename(src)
        dst = os.path.join(OUTPUT_DIR, name)

        with open(src) as f:
            items = [json.loads(line) for line in f if line.strip()]

        out = []
        changed = 0
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

    print()
    print("Done.")
    print(f"  Files:         {len(files)}")
    print(f"  Total lines:   {total_lines}")
    print(f"  Total changed: {total_changed} "
          f"({100 * total_changed / total_lines:.1f}%)" if total_lines else "")
    print(f"  Output:        {OUTPUT_DIR}")


if __name__ == "__main__":
    main()