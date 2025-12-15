#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, json, random, re, argparse, multiprocessing as mp
from functools import partial
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# ---------- Prompt-template helpers ----------
THOUGHT_RE  = re.compile(r"<\|begin_of_thought\|>\n\n(.*?)\n\n<\|end_of_thought\|>",  re.DOTALL)
SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>\n\n(.*?)\n\n<\|end_of_solution\|>", re.DOTALL)

TOKENIZER_PATH = "/checkpoint/lhu/models/OpenThinker2-7B"
tokenizer = None

def init_worker():
    """Initialise the global tokenizer once per worker."""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

def process_response(resp: str) -> str:
    """Replace <thought>/<solution> blocks with <think>… for assistant turns (if wanted)."""
    tm, sm = THOUGHT_RE.search(resp), SOLUTION_RE.search(resp)
    if not (tm and sm):
        return resp.strip()
    return f"<think>\n{tm.group(1).strip()}\n</think>\n\n{sm.group(1).strip()}"

# ---------- JSON-safe utility ----------
def to_json_safe(obj):
    """Recursively convert NumPy arrays to lists so json.dumps doesn’t choke (not used now, but handy)."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ---------- Build chat messages ----------
def build_messages(sample, *, use_think_format=False, use_system_prompt=False):
    msgs = []
    if use_system_prompt and "system" in sample:
        msgs.append({"role": "system", "content": sample["system"]})
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        if role == "assistant" and use_think_format:
            content = process_response(turn["value"])
        else:
            content = turn["value"]
        msgs.append({"role": role, "content": content})
    return msgs

def tokenize_full(sample, *, use_think_format=False, use_system_prompt=False):
    """Return list of token IDs for the *whole* sample conversation."""
    global tokenizer
    msgs = build_messages(sample,
                          use_think_format=use_think_format,
                          use_system_prompt=use_system_prompt)
    try:
        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0).tolist()
        return ids
    except Exception as e:
        print("⚠️  Tokenization error:", e)
        return None

# ---------- Worker wrapper ----------
def process_sample(sample,
                   *, use_think_format=False, use_system_prompt=False):
    ids = tokenize_full(sample,
                        use_think_format=use_think_format,
                        use_system_prompt=use_system_prompt)
    if ids is None:
        return None
    # Extract *first* user prompt string
    prompt_text = next(
        (turn["value"] for turn in sample["conversations"]
         if turn.get("from") == "user"), None)
    if prompt_text is None:
        return None
    return dict(prompt=prompt_text, n_tokens=len(ids))

# ---------- Main pipeline ----------
def main(input_path, output_path,
         *, bucket_size=50_000,
         use_think_format=True,
         use_system_prompt=False,
         n_workers=8):

    os.makedirs(output_path, exist_ok=True)

    # 1. Load data --------------------------------------------------------------------------------
    parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No *.parquet found under {input_path}")
    print(f"🗄️  Found {len(parquet_files)} parquet file(s). Reading...")

    dfs = [pd.read_parquet(p) for p in parquet_files]
    df  = pd.concat(dfs, ignore_index=True)
    samples = df.to_dict("records")
    print(f"✅  Loaded {len(samples):,} json-rows from parquets")

    # 2. Token-count each sample in parallel ------------------------------------------------------
    with mp.Pool(n_workers, initializer=init_worker) as pool:
        processed = list(
            tqdm(pool.imap(
                     partial(process_sample,
                             use_think_format=use_think_format,
                             use_system_prompt=use_system_prompt),
                     samples),
                 total=len(samples),
                 desc="Tokenising"))
    # Remove failures
    processed = [p for p in processed if p is not None]
    print(f"✅  Tokenised {len(processed):,} samples")

    # 3. Sort by token length ---------------------------------------------------------------------
    processed.sort(key=lambda x: x["n_tokens"])

    # 4. Slice into buckets of N prompts ----------------------------------------------------------
    print(f"📦  Bucketing {len(processed):,} prompts --> {bucket_size} prompts per file")
    for i in range(0, len(processed), bucket_size):
        bucket_idx = i // bucket_size
        bucket     = processed[i : i + bucket_size]

        # ----- compute stats -----
        tok_counts = [b["n_tokens"] for b in bucket]
        min_tok    = min(tok_counts)
        max_tok    = max(tok_counts)
        avg_tok    = int(round(sum(tok_counts) / len(tok_counts)))

        # ----- build filename -----
        out_fname = (
            f"bucket_{bucket_idx:04d}"
            f"_avg{avg_tok}_min{min_tok}_max{max_tok}.json"
        )
        out_path  = os.path.join(output_path, out_fname)

        # ----- write JSON array of prompt strings -----
        prompts = [" ".join(item["prompt"].split()) for item in bucket]  # tidy whitespace
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(prompts, fout, ensure_ascii=False, indent=2)

        print(f"- {out_fname}  ({len(prompts)} prompts, {sum(tok_counts):,} tokens)")


# ---------- Entry-point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bucket user prompts by token length. "
                    "Each output file contains a fixed NUMBER of prompts.")
    parser.add_argument("--input_path",  required=True,
                        help="Directory containing *.parquet files")
    parser.add_argument("--output_path", required=True,
                        help="Directory for bucketed prompt files")
    parser.add_argument("--bucket_size", type=int, default=5_000,
                        help="Number of prompts per output file (default: 5 000)")
    parser.add_argument("--n_workers",   type=int, default=8,
                        help="Tokenisation workers (default: 8)")
    parser.add_argument("--think_format",  action="store_true",
                        help="Apply <think> replacement to assistant messages")
    parser.add_argument("--system_prompt", action="store_true",
                        help="Include system field as first chat message")

    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    main(args.input_path,
         args.output_path,
         bucket_size=args.bucket_size,
         use_think_format=args.think_format,
         use_system_prompt=args.system_prompt,
         n_workers=args.n_workers)
