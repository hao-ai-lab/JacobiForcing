#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
import json
import random
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2ForCausalLM, AutoTokenizer

# ---------------------------
# Local import setup
# ---------------------------
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified import (
    jacobi_forward_greedy_multiblock
)
Qwen2ForCausalLM.jacobi_forward_greedy_multiblock = jacobi_forward_greedy_multiblock


# ---------------------------
# JSONL utilities
# ---------------------------
def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, save_path):
    with open(save_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def extract_python_code(text):
    match = re.search(r"```python([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("df_file", type=str, nargs="?",
                        default="/home/lah003/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
    parser.add_argument("model_name", type=str, nargs="?",
                        default="/raid/lah003/shiftedattn-10-16-7b-qwen2p5-coder-n32w16-n16distill-data-v2-ar-1-cyclic-noise-all-1e-6/ckpt-344092")
    parser.add_argument("tokenizer_name", type=str, nargs="?",
                        default="/home/lah003/models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("csv_path", type=str, nargs="?",
                        default="profiling_results/diffusion_profile_humaneval.csv")
    parser.add_argument("max_calls", type=int, nargs="?", default=1024)
    parser.add_argument("max_new_tokens", type=int, nargs="?", default=1024)

    # sweeping knobs
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--n_gram_pool_size", type=int, default=4)
    parser.add_argument("--lookahead_start_ratio", type=float, default=0.0)

    # eval
    parser.add_argument("--eval_dir", type=str,
                        default="/home/lah003/data/CLLM2_eval_generations/multiblock_testing_prompt")
    parser.add_argument("--original_jsonl", type=str, default="humaneval_python_example.jsonl")
    parser.add_argument("--out_prefix", type=str, default=None)

    return parser


def main():
    args = build_argparser().parse_args()

    n_token_seq_len = args.n_token_seq_len
    K = args.K
    r = args.r
    n_gram_pool_size = args.n_gram_pool_size
    lookahead_start_ratio = args.lookahead_start_ratio

    run_id = args.out_prefix or f"ntok{n_token_seq_len}_K{K}_r{r:.2f}_ng{n_gram_pool_size}"
    print(f"[RUN] {run_id}")

    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_parquet(args.df_file)
    df_size = len(df)
    print(f"Loaded dataset with {df_size} samples from {args.df_file}")
    records = df.to_dict(orient="records")

    # ---------------------------
    # Load model/tokenizer
    # ---------------------------
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model.eval()

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"eos id: {eos_id}")
    print(f"pad id: {pad_id}")

    max_calls = args.max_calls
    max_new_tokens = args.max_new_tokens

    # ---------------------------
    # Iterate dataset
    # ---------------------------
    all_rows = []
    all_generations = []
    total_gen_only_time = 0.0

    t0_overall = time.perf_counter()

    for idx, row in tqdm(enumerate(records), total=len(records)):
        task_id = row.get("task_id", f"idx_{idx}")

        prompt = """Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```
{}
```
""".strip().format(row["prompt"].strip())

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        input_ids = model_inputs["input_ids"]
        attention_mask = torch.full_like(input_ids, 1, device=model.device)

        # per-example stats
        iters = []
        total_new_tokens = 0
        calls = 0
        prev_len = input_ids.shape[1]
        prompt_len = prev_len
        stop_reason = None
        prefill_phase = True
        generated_ids = input_ids
        prefill_drafted_n_gram = None
        gen_only_time = 0.0

        t_start = time.time()

        while True:
            # Check EOS
            generated_part = generated_ids[0, prompt_len:]
            hit_eos = (generated_part == eos_id).any().item()

            if hit_eos:
                stop_reason = "eos"
                break
            if total_new_tokens >= max_new_tokens:
                stop_reason = "max_new_tokens"
                break
            if calls >= max_calls:
                stop_reason = "max_calls"
                break

            # One diffusion decoding call
            if prefill_phase:
                q_sampled = []
                for _ in range(n_token_seq_len):
                    q_sample = torch.tensor(
                        [random.choice(generated_ids[0].tolist())],
                        dtype=torch.long,
                        device=model.device,
                    ).unsqueeze(0)
                    q_sampled.append(q_sample)
                prefill_draft_token_ids = torch.cat(q_sampled, dim=1)

                prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

                past_key_values, first_correct_token, prefill_drafted_n_gram, itr_count = \
                    model.jacobi_forward_greedy_multiblock(
                        input_ids=prefill_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=True,
                        prefill_phase=True,
                        n_token_seq_len=n_token_seq_len,
                        K=K,
                        r=r,
                        n_gram_pool_size=n_gram_pool_size,
                        lookahead_start_ratio=lookahead_start_ratio,
                        tokenizer=tokenizer,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                    )

                prefill_phase = False
                generated_ids = input_ids  # committed prompt only

                # do not count prefill iterations
                itr_count = 0

            else:
                if calls == 1:
                    input_ids = prefill_drafted_n_gram
                else:
                    q_sampled = []
                    for _ in range(n_token_seq_len - 1):
                        q_sample = torch.tensor(
                            [random.choice(generated_ids[0].tolist())],
                            dtype=torch.long,
                            device=model.device,
                        ).unsqueeze(0)
                        q_sampled.append(q_sample)
                    q_sampled = torch.cat(q_sampled, dim=1)
                    input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)

                t_gen_start = time.perf_counter()
                past_key_values, first_correct_token, accepted_n_gram, itr_count = \
                    model.jacobi_forward_greedy_multiblock(
                        input_ids=input_ids,
                        attention_mask=None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        prefill_phase=False,
                        n_token_seq_len=n_token_seq_len,
                        K=K,
                        r=r,
                        n_gram_pool_size=n_gram_pool_size,
                        lookahead_start_ratio=lookahead_start_ratio,
                        tokenizer=tokenizer,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                    )
                gen_only_time += (time.perf_counter() - t_gen_start)
                generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

            calls += 1
            iters.append(int(itr_count))

            added = generated_ids.shape[1] - prev_len
            if added > 0:
                total_new_tokens += added
            prev_len = generated_ids.shape[1]

        # subtract prefill-token accounting
        total_new_tokens -= 1

        dt = time.time() - t_start
        total_iterations = sum(iters)

        avg_iter_per_call = (total_iterations / calls) if calls > 0 else float("nan")
        avg_iter_per_token = (total_iterations / total_new_tokens) if total_new_tokens > 0 else float("nan")
        toks_per_sec = (total_new_tokens / gen_only_time) if gen_only_time > 0 else float("nan")

        total_gen_only_time += gen_only_time

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_str = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False)
        all_generations.append(generated_str)

        all_rows.append({
            "index": idx,
            "task_id": task_id,
            "prompt_tokens": prompt_len,
            "new_tokens": total_new_tokens,
            "calls": calls,
            "total_iterations": total_iterations,
            "avg_iter_per_call": avg_iter_per_call,
            "avg_iter_per_token": avg_iter_per_token,
            "time_sec": dt,
            "toks_per_sec": toks_per_sec,
            "stop_reason": stop_reason,
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(records):
            print(f"====[{idx+1}/{len(records)}] task_id={task_id} "
                  f"new_toks={total_new_tokens} calls={calls} "
                  f"avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}====")

    # ---------------------------
    # Post-process generations (HumanEval)
    # ---------------------------#
    # TODO: support other datasets
    eval_dir = args.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    original_path = os.path.join(eval_dir, args.original_jsonl)
    original_generations = load_jsonl(original_path)

    for i, original_generation in enumerate(original_generations):
        original_generation["output"] = all_generations[i]
        processed_generation = extract_python_code(all_generations[i])
        original_generation["generation"] = processed_generation

    gen_save_path = os.path.join(
        eval_dir,
        f"{run_id}_humaneval_w_kv_generation_{Path(args.model_name).name}.jsonl"
    )
    save_jsonl(original_generations, gen_save_path)
    print(f"\n=== Generations saved to {gen_save_path} ===")

    # ---------------------------
    # Aggregate + save profiling CSV
    # ---------------------------
    t_overall = time.perf_counter() - t0_overall
    df_profile = pd.DataFrame(all_rows)

    # make sure csv dir exists
    csv_path = args.csv_path
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # if user provided a generic csv_path, prepend run_id automatically
    base_dir = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path)
    if base_name == "diffusion_profile_humaneval.csv":
        csv_path = os.path.join(base_dir, f"{run_id}_{base_name}")

    df_profile.to_csv(csv_path, index=False)
    print(f"Profiling CSV saved to {csv_path}")

    # EOS-only summary
    def _safe_mean(series):
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

    df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()

    print("\n=== Diffusion Decoding Profiling — EOS-only ===")
    print(f"Examples (eos): {len(df_eos)} / {len(df_profile)}   Total wall time: {t_overall:.4f}s")
    print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.4f}")
    print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.4f}")
    print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.4f}")
    print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.4f}")
    print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.4f}")

    print("\nStop reasons (all examples):")
    print(df_profile["stop_reason"].value_counts())

    eos_csv_path = csv_path.replace(".csv", "_eos.csv")
    df_eos.to_csv(eos_csv_path, index=False)
    print(f"EOS-only CSV saved to {eos_csv_path}")


if __name__ == "__main__":
    main()
