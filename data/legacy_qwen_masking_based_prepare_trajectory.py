from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import copy
from multiprocessing import Pool
import json
import math
import re

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def process_response(response: str) -> str:
    """
    Pull <|begin_of_thought|> … <|end_of_thought|> and
    <|begin_of_solution|> … <|end_of_solution|> sections out of the
    assistant response and wrap them in your <think> tag.
    """
    thought_match = re.search(
        r"<\|begin_of_thought\|>\n\n(.*?)\n\n<\|end_of_thought\|>",
        response,
        re.DOTALL,
    )
    solution_match = re.search(
        r"<\|begin_of_solution\|>\n\n(.*?)\n\n<\|end_of_solution\|>",
        response,
        re.DOTALL,
    )

    if not (thought_match and solution_match):
        # If the expected markers are missing, fall back to the raw text.
        return response.strip()

    thought = thought_match.group(1).strip()
    solution = solution_match.group(1).strip()
    return f"<think>\n{thought}\n</think>\n\n{solution}"


def build_messages(sample):
    """
    Convert a single OpenThoughts `sample` into the list-of-dict format
    expected by `apply_chat_template`.
    """
    messages = []
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        content = (
            turn["value"]
            if role == "user"
            else process_response(turn["value"])
        )
        messages.append({"role": role, "content": content})
    return messages


def process_sample(sample, tokenizer, chunk_size=32):
    messages = build_messages(sample)

    # --------------------------------------------------------------
    # 1.  Input prompt (everything up to *generation* start)
    #     `add_generation_prompt=True` adds an empty assistant turn
    #     according to the tokenizer’s template so the model knows
    #     where to begin generating.
    #
    # 2.  Full target text (includes assistant reply) used for the
    #     label / teacher-forcing trajectory.
    # --------------------------------------------------------------
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0).tolist()

    total_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,  # keep the assistant answer
        return_tensors="pt",
    ).squeeze(0).tolist()

    # Length guard
    if len(total_ids) > 16_384:
        return None

    # ----------------------------------------------------------
    # Build perturbation trajectory exactly as before
    # ----------------------------------------------------------
    trajectory = []
    for i in range(len(input_ids), len(total_ids), chunk_size):
        chunk_ids = total_ids[i : i + chunk_size]
        real_len = len(chunk_ids)

        batch_trajectory = []
        for j in range(real_len):
            modified = copy.deepcopy(chunk_ids)
            for k in range(j, real_len):
                # Replace future tokens with a random past token
                modified[k] = random.choice(total_ids[:i])
            batch_trajectory.append(modified)

        # Ground-truth chunk last
        batch_trajectory.append(chunk_ids)
        trajectory.append(batch_trajectory)

    return dict(
        sources_input_ids=input_ids,
        labels_ids=total_ids,
        trajectory=trajectory,
    )


def preprocess_openthoughts_parallel(
    data, tokenizer, batch_size=32, num_workers=16
):
    from functools import partial

    process_fn = partial(process_sample, tokenizer=tokenizer, chunk_size=batch_size)

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_fn, data), total=len(data)))

    return results


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        "/checkpoint/lhu/models/OpenThinker2-7B",
        trust_remote_code=True,
    )

    dataset = load_dataset(
        "parquet",
        data_files="/checkpoint/lhu/data/OpenThoughts-114k/data/train-00001-of-00006.parquet",
    )["train"]
    print("Dataset loaded:", len(dataset), "examples")

    chunk_size = 512
    num_chunks = math.ceil(len(dataset) / chunk_size)

    for i in range(num_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, len(dataset))
        print(f"Processing chunk {i+1}/{num_chunks} — rows {start}…{end-1}")

        sub_dataset = dataset.select(range(start, end))
        train_dataset = preprocess_openthoughts_parallel(
            sub_dataset,
            tokenizer,
            batch_size=32,
        )

        with open(
            "/checkpoint/lhu/data/CLLM2_openthought/train_openthoughts_chunk_2.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            for sample in train_dataset:
                if sample:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
