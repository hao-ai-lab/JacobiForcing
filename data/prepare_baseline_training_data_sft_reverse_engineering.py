#!/usr/bin/env python3
import argparse, json, sys
from typing import Any, Dict, List
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct labels_ids = prompt_ids + concat_of_all(last_j) from preprocessed JSONL."
    )
    p.add_argument("--input_path", required=True, help="Input JSONL from the previous stage.")
    p.add_argument("--output_path", required=True, help="Output JSONL with added labels_ids.")
    p.add_argument("--n-token-seq-length", type=int, required=True,
                   help="Same n used to build the pairs: each pair = sampled(n) + last(n).")
    p.add_argument("--best-effort", action="store_true",
                   help="If set, tolerate length mismatches by slicing whatever is available.")
    return p.parse_args()

def reconstruct_labels(entry: Dict[str, Any], n: int, best_effort: bool) -> List[int]:
    """
    Given one line's dict with:
      - prompt_ids: List[int]
      - complete_training_sequence_ids: List[int] = prompt_ids + concat over pairs of [sampled_kj | last_j]
      - traj_position_indices: List[int]  (length == number of pairs)
      - prompt_ids_len: int (optional; will verify if present)
    return labels_ids = prompt_ids + concat_of_all(last_j).
    """
    prompt: List[int] = entry["prompt_ids"]
    prompt_len_reported = entry.get("prompt_ids_len", len(prompt))

    full_seq: List[int] = entry["complete_training_sequence_ids"]
    tail_after_prompt = full_seq[len(prompt):]

    num_pairs = len(entry.get("traj_position_indices", []))
    expected_tail_len = 2 * n * num_pairs

    labels_tail: List[int] = []
    pos = 0
    for i in range(num_pairs):
        block = tail_after_prompt[pos:pos + 2 * n] if pos < len(tail_after_prompt) else []
        last_j = block[-n:] if len(block) >= n else block
        labels_tail.extend(last_j)
        pos += 2 * n

    labels_ids = list(prompt) + labels_tail
    return labels_ids

def main():
    args = parse_args()
    n = args.n_token_seq_length

    count_in, count_out = 0, 0
    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Reconstructing labels_ids"):
            line = line.strip()
            if not line:
                continue
            count_in += 1
            obj = json.loads(line)

            labels_ids = reconstruct_labels(obj, n=n, best_effort=args.best_effort)
            obj.pop("complete_training_sequence_ids")
            obj.pop("traj_position_indices")
            obj["labels_ids"] = labels_ids

            # obj["labels_ids_len"] = len(labels_ids)

            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")
            count_out += 1

    print(f"Read {count_in} lines, wrote {count_out} with labels_ids --> {args.output_path}")

if __name__ == "__main__":
    main()
