#!/usr/bin/env python3
import argparse, json, hashlib, random, os, sqlite3, tempfile, atexit, itertools, re
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm
from datasets import load_dataset

# -----------------------
# Args
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-RAM parallel preprocessing for efficient CLLM training JSONL."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n-token-seq-length", type=int, default=64)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 8))
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--single-process", action="store_true")
    return parser.parse_args()

# -----------------------
# Deterministic RNG
# -----------------------
def stable_seed(*parts: Any, base_seed: int = 0) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8")); h.update(b"|")
    h.update(base_seed.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "big", signed=False)

# -----------------------
# ID parsing helpers
# -----------------------
def parse_data_id_int(data_id: str) -> int:
    """Extract the integer from 'data_{id}' robustly."""
    if data_id.startswith("data_"):
        return int(data_id[5:])
    m = re.search(r"(\d+)", data_id)
    return int(m.group(1)) if m else 0

def parse_itr_int(itr_id: str) -> int:
    """Extract the integer from 'itr_{iteration_id}' robustly."""
    if itr_id.startswith("itr_"):
        return int(itr_id[4:])
    m = re.search(r"(\d+)", itr_id)
    return int(m.group(1)) if m else 0


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
    Entry contains *one* (k_j,last_j) pair; final stitching happens later.
    """
    data_id          = sample["data_id"]            # e.g., "data_123"
    diffusion_itr_id = sample["diffusion_itr_id"]   # e.g., "itr_7"
    data_id_int      = parse_data_id_int(data_id)
    diffusion_itr    = parse_itr_int(diffusion_itr_id)

    prompt_ids       = sample["prompt_ids"]
    answer_traj      = sample["answer_trajectory_ids"]

    if len(answer_traj) < 2:
        return None  # skip

    rng = random.Random(stable_seed(data_id, diffusion_itr_id, base_seed=base_seed))
    k_j = rng.randint(0, len(answer_traj) - 2)

    sampled_seq = answer_traj[k_j][-n_token_seq_length:]
    fixed_seq   = answer_traj[-1][-n_token_seq_length:]

    pair_seq = list(sampled_seq) + list(fixed_seq)
    
    entry = dict(
        data_id=data_id,
        data_id_int=int(data_id_int),
        prompt_ids=list(prompt_ids),
        pairs=[dict(
            diffusion_itr=int(diffusion_itr),  # for sorting
            traj_position_index=int(k_j),
            seq=pair_seq
        )],
    )
    return data_id, entry

# -----------------------
# In-memory merge helpers
# -----------------------
def merge_entry(existing: Dict[str, Any], new_entry: Dict[str, Any], verbose: bool = False):
    existing["pairs"].extend(new_entry["pairs"])
    if verbose:
        print(f"Merged duplicate data_id {existing['data_id']} "
              f"(total pairs: {len(existing['pairs'])})")

# -----------------------
# SQLite helpers
# -----------------------
def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            data_id     TEXT PRIMARY KEY,
            data_id_int INTEGER NOT NULL,
            value       TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_data_id_int ON entries(data_id_int)")
    return conn

def db_get(conn: sqlite3.Connection, data_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT value FROM entries WHERE data_id=?", (data_id,))
    row = cur.fetchone()
    return None if row is None else json.loads(row[0])

def db_put(conn: sqlite3.Connection, data_id: str, entry: Dict[str, Any]):
    conn.execute(
        "INSERT INTO entries (data_id, data_id_int, value) VALUES(?,?,?) "
        "ON CONFLICT(data_id) DO UPDATE SET value=excluded.value",
        (data_id, int(entry["data_id_int"]), json.dumps(entry, ensure_ascii=False))
    )

def merge_into_db(conn, data_id, entry, verbose):
    existing = db_get(conn, data_id)
    if existing is None:
        db_put(conn, data_id, entry)
    else:
        merge_entry(existing, entry, verbose)
        db_put(conn, data_id, existing)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    max_in_flight = args.max_in_flight or max(4, args.num_workers * 4)

    # Streaming dataset
    data = load_dataset(
        "json",
        data_files={"train": args.input_path},
        split="train",
        streaming=True,
        cache_dir=args.cache_dir
    )

    # SQLite store
    db_path = args.db_path or tempfile.mkstemp(
        prefix="merge_", suffix=".sqlite",
        dir=os.path.dirname(os.path.abspath(args.output_path)) or "."
    )[1]
    conn = open_db(db_path)
    atexit.register(lambda: conn.close())

    pbar = tqdm(desc="Processing", disable=args.no_progress)

    def handle(res):
        if res is None:
            pbar.update(1); return
        data_id, entry = res
        merge_into_db(conn, data_id, entry, args.verbose)
        if (pbar.n % 5000) == 0:
            conn.commit()
        pbar.update(1)

    try:
        if args.single_process or args.num_workers <= 1:
            for sample in data:
                handle(process_one_sample(sample, args.n_token_seq_length, args.seed))
            conn.commit()
        else:
            in_flight = set()
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                for i, sample in enumerate(data):
                    in_flight.add(ex.submit(process_one_sample,
                                            sample, args.n_token_seq_length, args.seed))
                    if len(in_flight) >= max_in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        for fut in done: handle(fut.result())
                for fut in as_completed(in_flight): handle(fut.result())
                conn.commit()
    finally:
        pbar.close()

    # -------- Final write-out with sorting --------
    with open(args.output_path, "w", encoding="utf-8") as fout:
        cur = cur = conn.execute("SELECT value FROM entries ORDER BY data_id_int")
        count = 0

        # across all data_id
        for (value_str,) in cur:
            entry = json.loads(value_str)

            # Sort the (k_j,last_j) pairs by diffusion_itr (int)
            pairs_sorted = sorted(entry["pairs"], key=lambda p: p["diffusion_itr"])

            # Flatten sequences
            concatenated_pairs: List[int] = list(
                itertools.chain.from_iterable(p["seq"] for p in pairs_sorted)
            )

            traj_position_indices: List[int] = list(
                p["traj_position_index"] for p in pairs_sorted
            )

            output_entry = dict(
                data_id               = entry["data_id"],
                prompt_ids            = entry["prompt_ids"][0],
                complete_training_sequence_ids = entry["prompt_ids"][0] + concatenated_pairs,
                prompt_ids_len = len(entry["prompt_ids"][0]),
                traj_position_indices = traj_position_indices,
            )
            fout.write(json.dumps(output_entry, ensure_ascii=False))
            fout.write("\n")
            count += 1

    # Remove temp DB if we created it
    if not args.db_path:
        try: os.remove(db_path)
        except Exception: pass

    print(f"Processed {count} unique data_id samples --> {args.output_path}")

if __name__ == "__main__":
    main()