#!/usr/bin/env python3
import argparse, json, os, random, sys

def detect_input_kind(path):
    # Return "jsonl" if not starting with '[', otherwise "json"
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return "jsonl"  # empty -> treat as jsonl
            if ch.isspace():
                continue
            return "json" if ch == "[" else "jsonl"

def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[warn] Skipping invalid JSON on line {lineno}: {e}\n")

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

def write_json_array(path, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def reservoir_sample(iterable, k, rng):
    """Return up to k items sampled uniformly without knowing the total size."""
    reservoir = []
    for i, item in enumerate(iterable, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = rng.randrange(i)
            if j < k:
                reservoir[j] = item
    return reservoir

def first_k(iterable, k):
    out = []
    for item in iterable:
        if len(out) >= k:
            break
        out.append(item)
    return out

def main():
    p = argparse.ArgumentParser(description="Downsample a JSONL or JSON array file to a specified size.")
    p.add_argument("--input", help="Input file path (JSONL or a JSON array).")
    p.add_argument("--output", help="Output file path.")
    p.add_argument("-n", "--size", type=int, required=True, help="Target number of records.")
    p.add_argument("--method", choices=["random", "first"], default="random", help="Sampling method.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for --method random).")
    p.add_argument("--output-format", choices=["jsonl", "json"], default="jsonl",
                   help="Output format (default: jsonl).")
    args = p.parse_args()

    if args.size <= 0:
        sys.stderr.write("[error] --size must be a positive integer.\n")
        sys.exit(2)

    rng = random.Random(args.seed)

    kind = detect_input_kind(args.input)

    # Read input as an iterator of dicts
    if kind == "json":
        data = load_json_array(args.input)
        if not isinstance(data, list):
            sys.stderr.write("[error] JSON input is not an array.\n")
            sys.exit(2)
        iterable = data
    else:
        iterable = read_jsonl(args.input)

    # Sample
    if args.method == "first":
        sampled = first_k(iterable, args.size)
    else:
        sampled = reservoir_sample(iterable, args.size, rng)

    # Write output
    if args.output_format == "json":
        write_json_array(args.output, sampled)
    else:
        write_jsonl(args.output, sampled)

    # Small summary to stderr
    sys.stderr.write(f"[ok] Wrote {len(sampled)} records to {args.output} "
                     f"({args.output_format}), method={args.method}, seed={args.seed}\n")

if __name__ == "__main__":
    main()
