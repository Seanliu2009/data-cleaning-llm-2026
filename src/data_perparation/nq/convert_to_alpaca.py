"""
Convert JSONL raw data (question/answer) to Alpaca format JSON array.
Input:  Directory containing .jsonl files with fields: 'question' and 'answer'
Output: Alpaca format .json files (instruction, input, output)
"""

import os
import json
import argparse
from pathlib import Path

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.paths import DATA_DIR

def convert_jsonl_to_alpaca(input_file, output_file, instruction_text):
    """
    Convert a single JSONL file to Alpaca JSON format.
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line in {input_file}: {e}")
                continue

            question = item.get('question', '')
            answer = item.get('answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else "I don't know."
            elif not answer:
                answer = "I don't know."
            else:
                answer = str(answer)

            alpaca_entry = {
                "instruction": instruction_text,
                "input": question,
                "output": answer
            }
            data.append(alpaca_entry)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)

    print(f"Converted {len(data)} samples from {os.path.basename(input_file)} to {os.path.basename(output_file)}")
    return len(data)

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL raw data to Alpaca format.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .jsonl files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save .json files')
    parser.add_argument('--instruction', type=str,
                        default="Answer the question based on your knowledge. If you don't know, respond with 'I don't know.'",
                        help='Instruction text for Alpaca format')
    parser.add_argument('--pattern', type=str, default='*.jsonl',
                        help='File pattern to match (default: *.jsonl)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_path = Path(args.input_dir)
    jsonl_files = list(input_path.glob(args.pattern))

    if not jsonl_files:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL file(s) to process.")

    total_samples = 0
    for jsonl_file in jsonl_files:
        output_filename = jsonl_file.stem + '.json'
        output_path = Path(args.output_dir) / output_filename
        count = convert_jsonl_to_alpaca(
            str(jsonl_file),
            str(output_path),
            args.instruction
        )
        total_samples += count

    print(f"\nAll conversions completed. Total samples processed: {total_samples}")

if __name__ == "__main__":
    main()