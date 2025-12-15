#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bucket **OpenCodeInstruct**-style data into fixed-count shards (default: 5k examples),
using a chat template to tokenize the full (user,input ↔ assistant,output) pair and
sorting by total token count.

Strict assumptions (per dataset card):
- Each JSONL row has fields: `input` (question/instruction) and `output` (LLM response).
- We emit ONLY `user` and `assistant` roles; no system role.
- Input format: ONLY `*.jsonl` under --input_path (one JSON object per line).
- Output: JSON files named `bucket_XXXX_avgA_minB_maxC.json`, each a JSON array of the
  FIRST user prompts (i.e., the `input` text) for the 5k examples in that bucket.

Usage:
    python bucket_opencodeinstruct.py \
        --input_path /path/to/jsonl_dir \
        --output_path /path/to/out \
        --tokenizer_path Qwen/Qwen2.5-7B-Instruct \
        --bucket_size 5000 \
        --n_workers 8
"""

import os, glob, json, argparse, multiprocessing as mp
from functools import partial
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_PATH: Optional[str] = "/checkpoint/lhu/models/Qwen2.5-Coder-7B-Instruct"  # set from CLI in main()
TOKENIZER = None                      # global per worker


def init_worker():
    """Initialise the global tokenizer once per worker."""
    global TOKENIZER
    if TOKENIZER is None:
        if not TOKENIZER_PATH:
            raise RuntimeError("TOKENIZER_PATH not set in worker.")
        TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


def tokenize_pair(user_text: str, assistant_text: str) -> Optional[int]:
    """Return total token count for a (user, assistant) pair via apply_chat_template."""
    global TOKENIZER
    msgs = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    try:
        ids = (
            TOKENIZER.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
            .squeeze(0)
            .tolist()
        )
        return len(ids)
    except Exception as e:
        print("⚠️  Tokenization error:", e)
        return None


def process_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build messages from `input`/`output`, tokenise, and return stats for bucketing."""
    inp = sample.get("input")
    out = sample.get("output")
    if not isinstance(inp, str) or not isinstance(out, str):
        return None
    n_tok = tokenize_pair(inp, out)

    # tidy whitespace of prompt for output file
    prompt_clean = " ".join(inp.split())
    return {"prompt": prompt_clean, "n_tokens": n_tok}


def load_all_records(input_path: str) -> List[Dict[str, Any]]:
    """Load only *.jsonl files under a directory into a list of dicts."""
    paths = sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No *.jsonl found under {input_path}")
    print(f"STEP 0:  Found {len(paths)} jsonl file(s). Reading...")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"⚠️  Skipping bad JSON line {i} in {os.path.basename(p)}: {e}")
                    continue
                rows.append(obj)
    print(f"STEP 1:  Loaded {len(rows):,} json-rows from {len(paths)} file(s)")
    return rows


def main(input_path: str,
         output_path: str,
         *,
         tokenizer_path: str,
         bucket_size: int = 5_000,
         n_workers: int = 8):

    global TOKENIZER_PATH
    TOKENIZER_PATH = tokenizer_path

    os.makedirs(output_path, exist_ok=True)

    # 1) Load ------------------------------------------------------------------------------------
    samples = load_all_records(input_path)

    # 2) Token-count each sample in parallel -----------------------------------------------------
    with mp.Pool(n_workers, initializer=init_worker) as pool:
        processed = list(
            tqdm(
                pool.imap(partial(process_sample), samples),
                total=len(samples),
                desc="Tokenising",
            )
        )

    processed = [p for p in processed if p is not None]
    print(f"STEP 2:  Tokenised {len(processed):,} samples")

    # 3) Sort by token length (ascending)
    processed.sort(key=lambda x: x["n_tokens"])

    # 4) Slice into buckets of N prompts
    print(f"STEP 3:  Bucketing {len(processed):,} prompts --> {bucket_size} prompts per file")
    for i in range(0, len(processed), bucket_size):
        bucket_idx = i // bucket_size
        bucket     = processed[i : i + bucket_size]
        if not bucket:
            continue

        tok_counts = [b["n_tokens"] for b in bucket]
        min_tok    = min(tok_counts)
        max_tok    = max(tok_counts)
        avg_tok    = int(round(sum(tok_counts) / len(tok_counts)))

        out_fname = (
            f"bucket_{bucket_idx:04d}"
            f"_avg{avg_tok}_min{min_tok}_max{max_tok}.json"
        )
        out_path  = os.path.join(output_path, out_fname)

        prompts = [item["prompt"] for item in bucket]
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(prompts, fout, ensure_ascii=False, indent=2)

        print(f"-- {out_fname}  ({len(prompts)} prompts, {sum(tok_counts):,} tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Bucket OpenCodeInstruct prompts (input→user) by total token length. "
            "Each output file contains a fixed NUMBER of prompts."
        )
    )
    parser.add_argument("--input_path",  required=True,
                        help="Directory containing *.jsonl files")
    parser.add_argument("--output_path", required=True,
                        help="Directory for bucketed prompt files")
    parser.add_argument("--tokenizer_path", required=True,
                        help="HF repo or local path for tokenizer with a chat template")
    parser.add_argument("--bucket_size", type=int, default=25_000,
                        help="Number of prompts per output file (default: 25 000)")
    parser.add_argument("--n_workers",   type=int, default=8,
                        help="Tokenisation workers (default: 8)")

    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    main(
        args.input_path,
        args.output_path,
        tokenizer_path=args.tokenizer_path,
        bucket_size=args.bucket_size,
        n_workers=args.n_workers,
    )
