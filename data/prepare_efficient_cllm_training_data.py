#!/usr/bin/env python3
import argparse
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
from datasets import load_dataset
import random
import os
import sqlite3
import tempfile
import atexit

# -----------------------
# Args
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-RAM parallel preprocessing for efficient CLLM training JSONL."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSON file.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file.")
    parser.add_argument("--n-token-seq-length", type=int, default=64,
                        help="Tail tokens to keep from each sequence block (default: 64).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF datasets cache dir (default: env HF_DATASETS_CACHE or None).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for deterministic per-sample RNG (default: 42).")
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 8),
                        help="Parallel worker processes (default: CPU count).")
    parser.add_argument("--max-in-flight", type=int, default=None,
                        help="Max tasks queued at once. Default: 4 * num-workers.")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to temporary SQLite DB for merging (default: output dir).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bar.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print merge details for duplicates.")
    parser.add_argument("--single-process", action="store_true",
                        help="Disable multiprocessing (lowest RAM, simpler I/O).")
    return parser.parse_args()

# -----------------------
# Deterministic RNG
# -----------------------
def stable_seed(*parts: Any, base_seed: int = 0) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    h.update(base_seed.to_bytes(8, byteorder="little", signed=False))
    return int.from_bytes(h.digest()[:8], "big", signed=False)

# -----------------------
# Per-sample transform
# -----------------------
def process_one_sample(
    sample: Dict[str, Any],
    n_token_seq_length: int,
    base_seed: int
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Return (data_id, entry) or None.
    """
    data_id = sample["data_id"]
    diffusion_itr_id = sample["diffusion_itr_id"]
    prompt_ids = sample["prompt_ids"]
    prompt_ids_len = sample["prompt_ids_len"]
    labels_ids = sample["labels_ids"]
    answer_trajectories = sample["answer_trajectory_ids"]

    if len(answer_trajectories) < 2:
        return None  # skip

    rng = random.Random(stable_seed(data_id, diffusion_itr_id, base_seed=base_seed))
    k_j = rng.randint(0, len(answer_trajectories) - 2)

    sampled_point_seq = answer_trajectories[k_j][-n_token_seq_length:]
    fixed_point_seq = answer_trajectories[-1][-n_token_seq_length:]

    #[p][k_0][last_0]...[k_j][last_j][k_{j+1}][last_{j+1}]...[k_T][last_T]
    # Avoid unnecessary copies; keep lists as they are
    complete_seq = list(prompt_ids) + list(sampled_point_seq) + list(fixed_point_seq)

    entry = dict(
        data_id=data_id,
        diffusion_itr_id=diffusion_itr_id,
        prompt_ids_len=prompt_ids_len,
        prompt_ids=list(prompt_ids),  # persisted once per data_id
        labels_ids=list(labels_ids),
        complete_training_sequence_ids=complete_seq,
        traj_position_indices=[int(k_j)],
    )
    return data_id, entry

# -----------------------
# In-memory merge (used by DB path)
# -----------------------
def merge_entry(existing: Dict[str, Any], new_entry: Dict[str, Any], verbose: bool = False):
    # Assume same prompt per data_id; use existing prompt length as split
    prompt_len = len(existing["prompt_ids"])
    additional = new_entry["complete_training_sequence_ids"][prompt_len:]
    existing["complete_training_sequence_ids"].extend(additional)
    existing["traj_position_indices"].extend(new_entry["traj_position_indices"])
    if verbose:
        print(f"Duplicate data_id {existing['data_id']} found, merging sequences.")
        print(f"additional sequence length: {len(additional)}")
        print(f"new total length: {len(existing['complete_training_sequence_ids'])}")

# -----------------------
# SQLite helpers
# -----------------------
def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            data_id TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        )
    """)
    return conn

def db_get(conn: sqlite3.Connection, data_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT value FROM entries WHERE data_id=?", (data_id,))
    row = cur.fetchone()
    if not row:
        return None
    return json.loads(row[0])

def db_put(conn: sqlite3.Connection, data_id: str, entry: Dict[str, Any]):
    # Upsert
    conn.execute(
        "INSERT INTO entries (data_id, value) VALUES (?, ?) "
        "ON CONFLICT(data_id) DO UPDATE SET value=excluded.value",
        (data_id, json.dumps(entry, ensure_ascii=False))
    )

def merge_into_db(conn: sqlite3.Connection, data_id: str, entry: Dict[str, Any], verbose: bool):
    existing = db_get(conn, data_id)
    if existing is None:
        db_put(conn, data_id, entry)
    else:
        merge_entry(existing, entry, verbose=verbose)
        db_put(conn, data_id, existing)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    max_in_flight = args.max_in_flight or max(4, args.num_workers * 4)

    # Dataset stream (does not load to RAM)
    data = load_dataset(
        "json",
        data_files={"train": args.input_path},
        split="train",
        streaming=True,
        cache_dir=args.cache_dir
    )

    # SQLite on-disk store for merges (keeps RAM tiny)
    if args.db_path:
        db_path = args.db_path
    else:
        # Put temp DB alongside output (faster than /tmp on some systems)
        out_dir = os.path.dirname(os.path.abspath(args.output_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        db_fd, db_path = tempfile.mkstemp(prefix="merge_", suffix=".sqlite", dir=out_dir)
        os.close(db_fd)  # sqlite will reopen; we just needed the path

    conn = open_db(db_path)
    atexit.register(lambda: conn.close())

    pbar = tqdm(desc="Processing", disable=args.no_progress)

    def handle_result(res):
        if res is None:
            pbar.update(1)
            return
        data_id, entry = res
        merge_into_db(conn, data_id, entry, verbose=args.verbose)
        # Commit in batches to keep OS cache happy
        if (pbar.n % 5000) == 0:
            conn.commit()
        pbar.update(1)

    try:
        if args.single_process or args.num_workers <= 1:
            # Lowest-RAM path: no multiprocessing, fully streaming
            for sample in data:
                res = process_one_sample(sample, args.n_token_seq_length, args.seed)
                handle_result(res)
            conn.commit()
        else:
            # Bounded in-flight multiprocessing
            in_flight = set()
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                # Feed tasks gradually, never letting the queue explode
                for sample in data:
                    in_flight.add(ex.submit(process_one_sample, sample, args.n_token_seq_length, args.seed))
                    if len(in_flight) >= max_in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        for fut in done:
                            handle_result(fut.result())

                # Drain remaining
                for fut in as_completed(in_flight):
                    handle_result(fut.result())
                conn.commit()
    finally:
        pbar.close()

    # Write JSONL directly from DB (streaming to disk; no big list in RAM)
    with open(args.output_path, "w", encoding="utf-8") as f:
        cur = conn.execute("SELECT value FROM entries ORDER BY data_id")
        count = 0
        for (value,) in cur:
            f.write(value)
            f.write("\n")
            count += 1

    # Clean up temp DB if user didn't supply a path
    if not args.db_path:
        try:
            conn.close()
        except Exception:
            pass
        try:
            os.remove(db_path)
        except Exception:
            pass

    print(f"Processed {count} unique data_id samples. Output saved to {args.output_path}")

if __name__ == "__main__":
    main()
