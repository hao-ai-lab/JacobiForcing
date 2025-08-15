#!/usr/bin/env python3
import argparse, json, os, sys, gzip
from typing import Iterable, Tuple, Union
from transformers import AutoTokenizer

Field = "complete_training_sequence_ids"

def open_maybe_gz(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")

def is_jsonl(path: str) -> bool:
    return path.endswith(".jsonl") or path.endswith(".jsonl.gz")

def load_json(path: str) -> list:
    with open_maybe_gz(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of records.")
    return data

def iter_jsonl(path: str) -> Iterable[str]:
    with open_maybe_gz(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def count_tokens(tokenizer, value: Union[list, str]) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        # count tokens without adding special tokens
        return len(tokenizer(value, add_special_tokens=False).input_ids)
    # Unknown type → treat as zero to keep the record
    return 0

def filter_stream_jsonl(in_path: str, out_path: str, tokenizer, field: str, threshold: int) -> Tuple[int, int]:
    kept = dropped = 0
    with open_maybe_gz(out_path, "w") as out_f:
        for line in iter_jsonl(in_path):
            obj = json.loads(line)
            tok_val = obj.get(field, None)
            n = count_tokens(tokenizer, tok_val) if tok_val is not None else 0
            if n > threshold:
                dropped += 1
            else:
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return kept, dropped

def filter_json_array(in_path: str, out_path: str, tokenizer, field: str, threshold: int) -> Tuple[int, int]:
    data = load_json(in_path)
    kept_data = []
    kept = dropped = 0
    for obj in data:
        tok_val = obj.get(field, None)
        n = count_tokens(tokenizer, tok_val) if tok_val is not None else 0
        if n > threshold:
            dropped += 1
        else:
            kept_data.append(obj)
            kept += 1
    with open_maybe_gz(out_path, "w") as f:
        json.dump(kept_data, f, ensure_ascii=False)
    return kept, dropped

def main():
    ap = argparse.ArgumentParser(description="Filter records by token length.")
    ap.add_argument("--input", help="Path to input .json / .jsonl (optionally .gz)")
    ap.add_argument("-o", "--output", help="Path to output (defaults to *_filtered.json/.jsonl)")
    ap.add_argument("--model", default="/checkpoint/lhu/models/OpenThinker2-7B", help="Tokenizer path")
    ap.add_argument("--field", default=Field, help="Field containing IDs or text")
    ap.add_argument("--threshold", type=int, default=14336, help="Max allowed tokens (strictly > is dropped)")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Derive output path if not provided
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(in_path)
        if ext == ".gz":
            base2, ext2 = os.path.splitext(base)
            ext = ext2 + ext  # e.g., .jsonl.gz
            base = base2
        out_path = f"{base}_filtered{ext or '.json'}"

    print(f"Loading tokenizer from {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f"Filtering '{in_path}' --> '{out_path}' using field '{args.field}' with threshold {args.threshold} tokens", file=sys.stderr)

    if is_jsonl(in_path):
        kept, dropped = filter_stream_jsonl(in_path, out_path, tokenizer, args.field, args.threshold)
    else:
        kept, dropped = filter_json_array(in_path, out_path, tokenizer, args.field, args.threshold)

    print(f"Done. kept={kept}, dropped={dropped}, total={kept + dropped}", file=sys.stderr)

if __name__ == "__main__":
    main()
