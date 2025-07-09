"""
––––––––––––––––––––––––––––––––––
final JSON format:

{
  "data_id":              str,                 # "data_<row-idx>"
  "diffusion_itr_id":     str,                 # always "itr_0" here
  "prompt_ids_len":       [int],               # list(len(prompt_ids))
  "prompt_ids":           [int, …],            # full encoded prompt
  "answer_trajectory_ids":[[int,…], …],        # 1 element per chunk
  "teacher_output_ids":   [int, …],            # = labels_ids
  "labels_ids":           [int, …]             # same as above (optional)
}
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random, copy, json, math, re, os
from multiprocessing import Pool
from functools import partial

THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>\n\n(.*?)\n\n<\|end_of_thought\|>", re.DOTALL
)
SOLUTION_RE = re.compile(
    r"<\|begin_of_solution\|>\n\n(.*?)\n\n<\|end_of_solution\|>", re.DOTALL
)

def process_response(resp: str) -> str:
    tm, sm = THOUGHT_RE.search(resp), SOLUTION_RE.search(resp)
    if not (tm and sm):
        return resp.strip()
    return f"<think>\n{tm.group(1).strip()}\n</think>\n\n{sm.group(1).strip()}"

def build_messages(sample):
    msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        content = turn["value"] if role == "user" else process_response(turn["value"])
        msgs.append({"role": role, "content": content})
    return msgs

def build_user_prompt(sample):
    msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
    for turn in sample["conversations"]:
        if turn["from"] == "user":
            role = "user"
            content = turn["value"]
        else:
            continue
        msgs.append({"role": role, "content": content})
    return msgs

def convert_sample(sample, row_id: int, tokenizer, chunk_size=32):
    """Return a list[dict] ready for the final JSON dump."""
    prompt_msgs = build_user_prompt(sample)
    msgs = build_messages(sample)

    # prompt (generation-prompt=True ➜ no assistant answer)
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, return_tensors="pt"
    ).squeeze(0).tolist()

    # target (= teacher output)
    full_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, return_tensors="pt"
    ).squeeze(0).tolist()

    print("prompt_ids:", len(prompt_ids), "tokens")
    print("full_ids:", len(full_ids), "tokens")

    if len(full_ids) > 16_384:    # length guard
        return []

    # build *answer_trajectory_ids*
    # keep only the ground-truth chunk for each 32-token block
    eos_id = tokenizer.eos_token_id

    # Number of tokens after the prompt
    response_length = len(full_ids) - len(prompt_ids)

    # If not divisible by chunk_size (32), pad with eos_id
    if response_length % chunk_size != 0:
        pad_amt = chunk_size - (response_length % chunk_size)
        full_ids = full_ids + [eos_id] * pad_amt
    
    answer_trajectory = []
    for i in range(len(prompt_ids), len(full_ids), chunk_size):
        chunk = full_ids[i : i + chunk_size]
        answer_trajectory.append(chunk)

    record = dict(
        data_id               = f"data_{row_id}",
        diffusion_itr_id      = "itr_0",
        prompt_ids_len        = [len(prompt_ids)],
        prompt_ids            = prompt_ids,
        answer_trajectory_ids = answer_trajectory,
        teacher_output_ids    = full_ids,
        labels_ids            = full_ids,           # set if --use_labels
    )
    assert ("answer_trajectory_ids" in record) and (record["answer_trajectory_ids"] is not None), f"Missing key in row {row_id}"

    return [record]

def preprocess_parallel(data, tokenizer, chunk_size=32, n_workers=16, start_idx=0):
    func = partial(convert_sample, tokenizer=tokenizer, chunk_size=chunk_size)

    # build (sample, row_id) pairs
    jobs = [(s, start_idx + i) for i, s in enumerate(data)]
    with Pool(n_workers) as pool:
        out = []
        for recs in tqdm(pool.starmap(func, jobs), total=len(jobs)):
            out.extend(recs)
    
    return out

if __name__ == "__main__":
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        "/checkpoint/lhu/models/OpenThinker2-7B", trust_remote_code=True
    )
    ds = load_dataset(
        "parquet",
        data_files="/checkpoint/lhu/data/OpenThoughts-114k/data/train-00000-of-00006.parquet",
    )["train"]
    print("Loaded", len(ds), "rows")

    CHUNK = 512            # rows per shard you want on disk
    outfile = "/checkpoint/lhu/data/CLLM2_openthought/train_openthoughts_chunk_0_formatted.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    all_records = []
    for shard in range(math.ceil(len(ds) / CHUNK)):
        a, b = shard * CHUNK, min((shard + 1) * CHUNK, len(ds))
        print(f"Processing rows {a}…{b-1}")
        sub = ds.select(range(a, b))
        all_records.extend(
            preprocess_parallel(sub, tokenizer, chunk_size=32, start_idx=a)
        )

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(all_records, f)

    print(f"Wrote {len(all_records):,} aligned records ➜ {outfile}")
