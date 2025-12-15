from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random, json, math, re, os
import multiprocessing as mp
from functools import partial
from glob import glob

# =====================
# Tokenizer (global per process)
# =====================
tokenizer = None
# Use the same tokenizer as your generation script for consistent chat templates
TOKENIZER_PATH = "/home/lah003/models/Qwen2.5-Coder-7B-Instruct"


def init_worker():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


# =====================
# Helpers
# =====================

def build_messages(sample, use_system_prompt=False):
    if use_system_prompt and sample.get("system"):
        msgs = [{"role": "system", "content": sample["system"]}]
    else:
        msgs = []
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        content = turn["value"]
        msgs.append({"role": role, "content": content})
    return msgs


def build_user_prompt(sample, use_system_prompt=False):
    if use_system_prompt and sample.get("system"):
        msgs = [{"role": "system", "content": sample["system"]}]
    else:
        msgs = []
    for turn in sample["conversations"]:
        if turn["from"] == "user":
            msgs.append({"role": "user", "content": turn["value"]})
    return msgs


def corrupt_chunk(chunk, i, full_ids, prompt_ids_len, lookup_context_len, pad_id):
    """
    Return list of progressively corrupted versions of the chunk,
    where each is a full-length accepted tokens up to end of this chunk,
    with rightmost n tokens of this chunk replaced by random context tokens.
    """
    corrupted_sequence = []
    start_idx = prompt_ids_len + i
    prefix = full_ids[:start_idx]
    chunk_len = len(chunk)
    for corrupt_right in reversed(range(chunk_len + 1)):
        keep = chunk[: chunk_len - corrupt_right] if corrupt_right > 0 else chunk[:]
        # corrupt these tokens with random tokens from context
        corrupt = []
        if corrupt_right > 0:
            sampling_start = max(0, start_idx - lookup_context_len)
            sampling_pool = full_ids[sampling_start:start_idx]
            if not sampling_pool:
                sampling_pool = [pad_id]
            corrupt = [random.choice(sampling_pool) for _ in range(corrupt_right)]
        # prefix + (corrupted chunk)
        seq = prefix + keep + corrupt
        corrupted_sequence.append(seq)
    return corrupted_sequence


def convert_sample(
    sample,
    row_id: int,
    chunk_size=32,
    use_system_prompt=False,
    lookup_context_len=128,
    sequence_sampling_ratio=1.0,
):
    global tokenizer

    try:
        prompt_msgs = build_user_prompt(sample, use_system_prompt=use_system_prompt)
        msgs = build_messages(sample, use_system_prompt=use_system_prompt)

        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).squeeze(0).tolist()

        full_ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0).tolist()

        pad_id = tokenizer.eos_token_id

        # Skip overly long examples
        if len(full_ids) > 16_384:
            return []

        response_length = len(full_ids) - len(prompt_ids)
        if response_length % chunk_size != 0:
            pad_amt = chunk_size - (response_length % chunk_size)
            full_ids = full_ids + [pad_id] * pad_amt

        records = []
        num_chunks = (len(full_ids) - len(prompt_ids)) // chunk_size

        # === Sampling indices to include according to the ratio ===
        chunk_indices = list(range(num_chunks))
        num_to_sample = max(1, int(num_chunks * sequence_sampling_ratio))  # always keep at least 1
        sampled_indices = set(random.sample(chunk_indices, num_to_sample))
        # =========================================================

        for chunk_idx in range(num_chunks):
            if chunk_idx not in sampled_indices:
                continue

            i = chunk_idx * chunk_size
            chunk = full_ids[len(prompt_ids) + i : len(prompt_ids) + i + chunk_size]
            answer_trajectory = corrupt_chunk(
                chunk, i, full_ids, len(prompt_ids), lookup_context_len, pad_id
            )
            record = dict(
                data_id=f"data_{row_id}",
                diffusion_itr_id=f"itr_{chunk_idx}",
                prompt_ids_len=[len(prompt_ids)],
                prompt_ids=prompt_ids,
                answer_trajectory_ids=answer_trajectory,
                teacher_output_ids=full_ids,
                labels_ids=full_ids,
            )
            records.append(record)

        return records
    except Exception as e:
        import traceback
        print(f"❌ Worker crashed on row {row_id}: {e}")
        traceback.print_exc()
        return []


def preprocess_parallel(
    data,
    chunk_size=32,
    n_workers=4,
    start_idx=0,
    use_system_prompt=False,
    lookup_context_len=128,
    sequence_sampling_ratio=1.0,
):
    func = partial(
        convert_sample,
        chunk_size=chunk_size,
        use_system_prompt=use_system_prompt,
        lookup_context_len=lookup_context_len,
        sequence_sampling_ratio=sequence_sampling_ratio,
    )
    jobs = [(s, start_idx + i) for i, s in enumerate(data)]
    with mp.Pool(n_workers, initializer=init_worker, maxtasksperchild=200) as pool:
        out = []
        for recs in tqdm(pool.starmap(func, jobs), total=len(jobs)):
            out.extend(recs)
    return out


# =====================
# OpenCodeInstruct -> training samples
# =====================

def opencode_row_to_sample(row: dict) -> dict:
    """Use the dataset's `input` as the user prompt and `output` as the
    assistant reply. No system message by default.
    """
    user_text = (row.get("input") or "").strip()
    assistant_text = (row.get("output") or "").strip()

    return {
        "system": None,
        "conversations": [
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
        "meta": {"id": row.get("id"), "domain": row.get("domain")},
    }


if __name__ == "__main__":
    random.seed(42)
    
    mp.set_start_method("spawn", force=True)

    # =========================================
    # Load OpenCodeInstruct from local Parquet shards
    DATA_PATH = "/home/lah003/data/CLLM2_preproc_opencodeinstruct/filtered/filtered_opencodeinstruct.jsonl"
    # ========================================= 
    
    if os.path.isdir(DATA_PATH):
        files = sorted(glob(os.path.join(DATA_PATH, "**/*.parquet"), recursive=True))
        print(f"Found {len(files)} parquet shards under {DATA_PATH}")
        ds = load_dataset("parquet", data_files=files, split="train")
        print("Loaded dataset from parquets with", len(ds), "rows")
    elif os.path.isfile(DATA_PATH) and DATA_PATH.endswith(".jsonl"):
        print(f"Loading dataset from JSONL: {DATA_PATH}")
        ds = load_dataset("json", data_files=DATA_PATH, split="train")
        print("Loaded dataset from JSONL with", len(ds), "rows")
    else:
        raise FileNotFoundError(f"DATA_PATH {DATA_PATH} is neither a directory of parquets nor a JSONL file")

    # Optionally shuffle/select subset (same pattern as before)
    SPLIT_RATIO = 3
    subset_size = len(ds) // SPLIT_RATIO
    indices = list(range(len(ds)))
    random.shuffle(indices)
    selected_indices = indices[subset_size:subset_size*2]
    ds_subset = ds.select(selected_indices)

    print("Subset size:", len(ds_subset))

    # ---------------------------
    # Preprocess
    # ---------------------------
    CHUNK = 1024
    N_TOKEN_SEQ_LENGTH = 32
    LOOKUP_CONTEXT_LENGTH = N_TOKEN_SEQ_LENGTH * 10
    SEQUENCE_SAMPLING_RATIO = 1

    OUTFILE_DIR = "/home/lah003/data/CLLM2_preproc_opencodeinstruct/pseudo_trajectories"
    os.makedirs(OUTFILE_DIR, exist_ok=True)
    OUTFILE = os.path.join(
        OUTFILE_DIR,
        f"idx_1_train_opencodeinstruct_split_ratio_{SPLIT_RATIO}_size_{len(ds_subset)}_ntok_size_{N_TOKEN_SEQ_LENGTH}_lookup_size_{LOOKUP_CONTEXT_LENGTH}_sampling_ratio_{SEQUENCE_SAMPLING_RATIO}_eos_tokens_termination_without_think_format_without_sysmsg.json",
    )

    N_WORKERS = min(12, os.cpu_count() or 4)

    with open(OUTFILE, "w", encoding="utf-8") as f:
        for shard in range(math.ceil(len(ds_subset) / CHUNK)):
            a, b = shard * CHUNK, min((shard + 1) * CHUNK, len(ds_subset))
            print(f"Processing rows {a}…{b-1}")
            sub = ds_subset.select(range(a, b))

            # Convert to list of *conversation* dicts
            sub_rows = [dict(x) for x in sub]
            sub_samples = [opencode_row_to_sample(r) for r in sub_rows]

            records = preprocess_parallel(
                sub_samples,
                chunk_size=N_TOKEN_SEQ_LENGTH,
                n_workers=N_WORKERS,
                start_idx=a,
                use_system_prompt=False,
                lookup_context_len=LOOKUP_CONTEXT_LENGTH,
                sequence_sampling_ratio=SEQUENCE_SAMPLING_RATIO,
            )
            for rec in records:
                f.write(json.dumps(rec) + "\n")

            print(f"Wrote {len(records):,} aligned records --> {OUTFILE}")
            del records, sub_rows, sub_samples

    print("\nFinished. Output:", OUTFILE)
