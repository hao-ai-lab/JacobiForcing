import os, glob, math, json, random, re
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import multiprocessing as mp
from functools import partial
import pandas as pd

import numpy as np

# Prompt regex as in your template
THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>\n\n(.*?)\n\n<\|end_of_thought\|>", re.DOTALL)
SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>\n\n(.*?)\n\n<\|end_of_solution\|>", re.DOTALL)

# Tokenizer location
TOKENIZER_PATH = "/checkpoint/lhu/models/OpenThinker2-7B"
tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

def process_response(resp: str) -> str:
    tm, sm = THOUGHT_RE.search(resp), SOLUTION_RE.search(resp)
    if not (tm and sm):
        return resp.strip()
    return f"<think>\n{tm.group(1).strip()}\n</think>\n\n{sm.group(1).strip()}"

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def build_messages(sample, use_think_format=False, use_system_prompt=False):
    msgs = []
    if use_system_prompt:
        msgs.append({"role": "system", "content": sample["system"]})
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        content = turn["value"] if role == "user" else process_response(turn["value"]) if use_think_format else turn["value"]
        msgs.append({"role": role, "content": content})
    return msgs

def tokenize_full_ids(sample, use_think_format=False, use_system_prompt=False):
    global tokenizer
    msgs = build_messages(sample, use_think_format=use_think_format, use_system_prompt=use_system_prompt)
    try:
        full_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).squeeze(0).tolist()
        return full_ids
    except Exception as e:
        print("Tokenization error:", e)
        return None

def process_sample(row, use_think_format=False, use_system_prompt=False):
    full_ids = tokenize_full_ids(row, use_think_format=use_think_format, use_system_prompt=use_system_prompt)
    if full_ids is None:
        return None
    return {
        "row": row,
        "full_ids": full_ids,
        "n_tokens": len(full_ids)
    }

def main(input_path, output_path, use_think_format=True, use_system_prompt=False, bucket_size=50000, n_workers=8):
    os.makedirs(output_path, exist_ok=True)

    # Gather all parquet files
    parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    assert len(parquet_files) > 0, f"No parquets found in {input_path}"

    # Load and concat all parquets using pandas for speed
    dfs = []
    for f in parquet_files:
        print(f"Loading {f} ...")
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    data = df.to_dict("records")
    print(f"Total samples: {len(data)}")

    # Multiprocess tokenization and count
    with mp.Pool(n_workers, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap(partial(process_sample, use_think_format=use_think_format, use_system_prompt=use_system_prompt), data),
                total=len(data)
            )
        )
    # Remove failed tokenizations
    samples = [r for r in results if r is not None]
    print(f"Tokenized samples: {len(samples)}")

    # Sort by token count
    samples.sort(key=lambda x: x["n_tokens"])

    # Bucket into 50k token splits
    bucket, cur_tokens, bucket_id = [], 0, 0
    for entry in samples:
        if cur_tokens + entry["n_tokens"] > bucket_size and len(bucket) > 0:
            out_file = os.path.join(output_path, f"bucket_{bucket_id:04d}.jsonl")
            print(f"Writing bucket {bucket_id} ({len(bucket)} samples, {cur_tokens} tokens) --> {out_file}")
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in bucket:
                    f.write(json.dumps(to_json_safe(rec["row"]), ensure_ascii=False) + "\n")

            bucket, cur_tokens = [], 0
            bucket_id += 1
        bucket.append(entry)
        cur_tokens += entry["n_tokens"]

    # Write the last bucket
    if len(bucket) > 0:
        out_file = os.path.join(output_path, f"bucket_{bucket_id:04d}.jsonl")
        print(f"Writing bucket {bucket_id} ({len(bucket)} samples, {cur_tokens} tokens) --> {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in bucket:
                f.write(json.dumps(to_json_safe(rec["row"]), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Directory with input parquets")
    parser.add_argument("--output_path", required=True, help="Directory to dump bucketed output")
    parser.add_argument("--bucket_size", type=int, default=50000, help="Token count per bucket")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--think_format", action="store_true")
    parser.add_argument("--system_prompt", action="store_true")

    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    main(
        args.input_path,
        args.output_path,
        use_think_format=args.think_format,
        use_system_prompt=args.system_prompt,
        bucket_size=args.bucket_size,
        n_workers=args.n_workers,
    )
